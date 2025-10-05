from __future__ import annotations
import os, shlex, json, uuid, threading, queue, subprocess, sqlite3
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel


CLIPS_ROOT = os.environ.get("CLIPS", "./clips")   # Also watch but store here
DATA_ROOT = os.path.join(CLIPS_ROOT, "data")
WEB_ROOT = os.path.join(DATA_ROOT, "web")
DB_PATH = os.path.join(DATA_ROOT, "ClipThing.db")

# derived assets go into DATA_ROOT
DATA_THUMBS = os.path.join(DATA_ROOT, "thumbnails")
DATA_PROXIES = os.path.join(DATA_ROOT, "proxies")
DATA_EXPORTS = os.path.join(DATA_ROOT, "exports")

for d in (CLIPS_ROOT, WEB_ROOT, DATA_THUMBS, DATA_PROXIES, DATA_EXPORTS):
    os.makedirs(d, exist_ok=True)

# --- Job definitions ---
@dataclass(order=True)
class PriorityItem:
    priority: int
    UUID: str = field(compare=False)
    start: Optional[float] = field(compare=False, default=None)
    end: Optional[float] = field(compare=False, default=None)
    # trim: Tuple[float, float] = field(compare=False)
    volumes: Optional[List[float]] = field(compare=False, default=None)
    export_limit_mb: Optional[float] = field(compare=False, default=None)

# --- queue ---
jobsQueue = queue.PriorityQueue()

# --- Init DB ---
def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS clips (
        uuid TEXT PRIMARY KEY,
        filename TEXT,
        creation_time REAL,
        size_bytes INTEGER,
        audio_tracks INTEGER,
        duration REAL,
        edited_volumes TEXT,
        edited_start REAL,
        edited_stop REAL,
        edited_title TEXT,
        exported_values TEXT
    ) WITHOUT ROWID;
    """)
    conn.commit()
    conn.close()

def init_scan():
    existing_files = set()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()
    rows = cur.execute("SELECT filename FROM clips").fetchall()
    for r in rows:
        existing_files.add(r[0])
    conn.close()

    for f in os.listdir(CLIPS_ROOT):
        if f in existing_files:
            continue
        if not f.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv')):
            continue
        new_uuid = str(uuid.uuid4())
        src_path = os.path.join(CLIPS_ROOT, f)
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute("""
        INSERT INTO clips (uuid, filename) VALUES (?, ?)
        """, (new_uuid, os.path.basename(src_path)))
        conn.commit()
        conn.close()
        jobsQueue.put(PriorityItem(10, new_uuid))  # Metadata
        jobsQueue.put(PriorityItem(20, new_uuid))  # Proxy
        jobsQueue.put(PriorityItem(30, new_uuid))  # Thumbnail
        print(f"🇬🇧 Discovered new clip: {f} as {new_uuid}")

def db_list_all_clips() -> List[dict]:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT uuid, creation_time, size_bytes, duration, edited_title, filename FROM clips ORDER BY creation_time DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

# --- Helpers ---
def ffprobe_duration(path: str) -> float:
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(path)}'
    out = subprocess.check_output(cmd, shell=True, text=True).strip()
    return float(out)

def compute_bitrates(size_limit_mb, duration, audio_kbps):
    size_bytes = int(size_limit_mb * 1024 * 1024)
    total_bps = (size_bytes * 8) / max(0.1, duration)
    audio_bps = max(64_000, int(audio_kbps * 1000))
    video_bps = int(max(100_000, total_bps - audio_bps))
    return video_bps, audio_bps

def ffprobe_trackcount(file_path: str) -> int:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "json",
        file_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    
    data = json.loads(result.stdout)
    indexes = [stream["index"] for stream in data.get("streams", [])]
    return len(indexes)

def worker_loop():
    jobsQueue.join()
    print("Worker thread started")
    while True:
        print("Worker waiting for job...")
        # try:
        item = jobsQueue.get()
        print(f"Worker got job: {item} with priority {item.priority}")
        match item.priority:

            case 1:  # Export
                # spawn ffmpeg, needed vars are file (deduce from uuid), start, end, volumes, size limit
                export_uuid = item.UUID
                # use DB to locate source video
                conn = sqlite3.connect(DB_PATH)
                r = conn.execute("SELECT filename, edited_title FROM clips WHERE uuid=?", (export_uuid,)).fetchone()
                conn.close()
                if not r:
                    raise FileNotFoundError("clip not found")
                input_file = os.path.join(CLIPS_ROOT, r[0])
                
                edited_title = r[1] if r[1] else export_uuid
                exported_file = os.path.join(DATA_EXPORTS, f"{edited_title}.mp4")

                export_size_limit = item.export_limit_mb
                export_volumes = item.volumes
                export_start = item.start
                export_end = item.end

                # Bitrates
                duration = max(0.1, export_end - export_start)
                export_video_bps, export_audio_bps = compute_bitrates(export_size_limit, duration, 160) ## hardcoded audio_kbps !!!

                #filter complex logic
                filters = []
                map_inputs = []
                for index, vol in enumerate(export_volumes):
                    filters.append(f"[0:a:{index}]volume={vol}[a{index}]")
                    map_inputs.append(f"[a{index}]")
                amix = "".join(map_inputs) + f"amix=inputs={len(export_volumes)}:duration=longest[mixed]"
                audio_filter = "; ".join(filters + [amix])
                #end of the horror

                # Pass 1
                pass1 = ["ffmpeg",
                                "-y", "-ss", str(export_start),
                                "-t", str(duration), "-i", input_file,
                                "-an", #skip audio
                                "-c:v", "libx264", "-preset", "medium", "-b:v", export_video_bps,
                                "-pass", "1", "-f", "mp4", os.devnull
                ]
                p = subprocess.Popen(pass1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                p.communicate()
                
                # Pass 2
                pass2 = ["ffmpeg",
                                "-y", "-ss", str(export_start), "-t", str(duration),
                                "-i", input_file,
                                "-filter_complex", audio_filter,
                                "-map", "[mixed]",
                                "-c:v", "libx264", "-preset", "medium", "-b:v", export_video_bps,
                                "-c:a", "aac", "-b:a", export_audio_bps,
                                "-movflags", "+faststart",
                                "-pass", "2",  exported_file
                ]
                p = subprocess.Popen(pass2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                p.communicate()

                jobsQueue.task_done()
                continue

            case 10: # Metadata

                uuid = item.UUID
                conn = sqlite3.connect(DB_PATH)
                r = conn.execute("SELECT filename FROM clips WHERE uuid=?", (uuid,)).fetchone()
                conn.close()
                if not r:
                    print("file not found")
                    jobsQueue.task_done()
                    return
                
                filepath = os.path.join(CLIPS_ROOT, r[0])
                
                creation_date = os.path.getmtime(filepath)
                size_bytes = os.path.getsize(filepath)
                try:
                    audio_tracks = ffprobe_trackcount(filepath)
                except Exception:
                    audio_tracks = None

                duration = ffprobe_duration(filepath)

                conn = sqlite3.connect(DB_PATH)
                conn.execute("""
                UPDATE clips SET
                    creation_time=?,
                    size_bytes=?,
                    audio_tracks=?,
                    duration=?
                WHERE uuid=?
                """, (creation_date, size_bytes, audio_tracks, duration, uuid))
                conn.commit()
                conn.close()
                jobsQueue.task_done()
                continue

            case 20: # Proxy

                uuid = item.UUID

                conn = sqlite3.connect(DB_PATH)
                r = conn.execute("SELECT filename FROM clips WHERE uuid=?", (uuid,)).fetchone()
                conn.close()
                if not r:
                    print("file not found")
                    jobsQueue.task_done()
                    return

                filepath = os.path.join(CLIPS_ROOT, r[0])

                proxy_path = os.path.join(DATA_PROXIES, f"{uuid}.mp4")
                cmd = (f"ffmpeg -y -i {shlex.quote(filepath)} -map 0:v:0 -map 0:a:0 "
                        f"-c:v libx264 -preset veryfast -crf 23 -vf scale='min(1280\\,iw)':-2 "
                        f"-c:a aac -b:a 128k -movflags +faststart {shlex.quote(proxy_path)}")
                subprocess.call(cmd, shell=True)
                jobsQueue.task_done()
                continue

            case 30: # Thumbnail

                uuid = item.UUID

                conn = sqlite3.connect(DB_PATH)
                r = conn.execute("SELECT filename, duration FROM clips WHERE uuid=?", (uuid,)).fetchone()
                conn.close()

                if not r:
                    print("file not found")
                    jobsQueue.task_done()
                    return
                filepath = os.path.join(CLIPS_ROOT, r[0])
                thumb_path = os.path.join(DATA_THUMBS, f"{uuid}.jpg")

                dur = r[1] if r[1] is not None else 2.0 ## ok that was faster and easier than expected

                t = max(0.0, dur / 2.0)
                os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
                thumbnail_ffmpeg = f'ffmpeg -y -ss {t:.2f} -i {shlex.quote(filepath)} -update true -frames:v 1 -vf "scale=min(480\\,iw):-2" {shlex.quote(thumb_path)}'

                subprocess.check_call(thumbnail_ffmpeg, shell=True)

                jobsQueue.task_done()
                continue

            case _:
                print(f"Unknown priority {item.priority}")
                jobsQueue.task_done()
                continue
        # except Exception as e:
        #     print(f"Error in worker loop: {e}")
        # finally:
        #     jobsQueue.task_done()

threading.Thread(target=worker_loop, daemon=True).start()




# --- API ---
app = FastAPI(title="Clipper")

@app.get("/clips")
def list_clips():
    rows = db_list_all_clips()
    return [{
        "uuid": r["uuid"],
        "file": r["filename"],
        "size_bytes": r["size_bytes"],
        "edited_title": r["edited_title"],
        "creation_time": r["creation_time"]
    } for r in rows]

@app.get("/clips/{UUID}/thumb")
def clip_thumb(UUID: str):
    conn = sqlite3.connect(DB_PATH)
    r = conn.execute("SELECT filename FROM clips WHERE uuid=?", (UUID,)).fetchone()
    conn.close()
    if not r: raise HTTPException(404)
    thumb_path = os.path.join(DATA_THUMBS, f"{UUID}.jpg")
    return FileResponse(thumb_path, media_type="image/jpeg")

@app.get("/clips/{UUID}/play")
def play_proxy(UUID: str):
    proxy_path = os.path.join(DATA_PROXIES, f"{UUID}.mp4")
    if not os.path.exists(proxy_path):
        raise HTTPException(404)
    return FileResponse(proxy_path, media_type="video/mp4")

class ExportValues(BaseModel):
    start: float
    end: float
    volumes: List[float]
    size_limit_mb: float = 50.0

@app.post("/clips/{UUID}/export")
def enqueue_export(UUID: str, body: ExportValues, status_code=202):

    exportedValues = json.dumps(asdict(body))

    conn = sqlite3.connect(DB_PATH)
    r = conn.execute("""UPDATE clips SET
        exported_values=? 
    WHERE uuid=?
    """, (exportedValues, UUID))
    conn.close()

    if not r: raise HTTPException(404)

    jobsQueue.put(PriorityItem(
        1, UUID, body.start, body.end, 
        body.volumes, body.size_limit_mb
        ))

    return 

@app.get("/clips/{UUID}/exported_file")
def serve_export_file(UUID: str):
    path = os.path.join(DATA_EXPORTS, f"{UUID}.mp4")
    if not os.path.exists(path): raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4")

@app.get("/queueSize")
def queueSize():
    if not jobsQueue.qsize():
        return {"queueSize": 0}
    return {"queueSize": jobsQueue.qsize()}

@app.get("/", response_class=HTMLResponse)
def root():
    index = os.path.join(WEB_ROOT, "index.html")
    return FileResponse(index) if os.path.exists(index) else HTMLResponse("Missing index.html")

@app.get("/missingno.jpg", response_class=HTMLResponse)
def missingno():
    missingno_path = os.path.join(WEB_ROOT, "missingno.jpg")
    if os.path.exists(missingno_path):
        return FileResponse(missingno_path, media_type="image/jpeg")
    else:
        raise HTTPException(404)
    

# --- Startup ---
init_db()
init_scan()

# print(jobsQueue.qsize())