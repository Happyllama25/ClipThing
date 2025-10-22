from __future__ import annotations
from dataclasses import dataclass, field
import webbrowser
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
# from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction
# from PyQt5.QtGui import QIcon
from sanitize_filename import sanitize
from typing import Optional, List
import json
import os
import queue
import shlex
import sqlite3
import subprocess
import sys
import threading
import uuid
import uvicorn

HOME_DIR = os.path.expanduser("~")
CLIPS_ROOT = os.environ.get("CLIPS", f"{HOME_DIR}/Videos/clips")

DATA_ROOT = os.path.join(CLIPS_ROOT, "data")
DATA_EXPORTS = os.path.join(CLIPS_ROOT, "exports")

#should we make a logs folder to contain ffmpeg logs? maybe futureproofing for multiple workers simultaneosly
DB_PATH = os.path.join(DATA_ROOT, "ClipThing.db")
DATA_THUMBS = os.path.join(DATA_ROOT, "thumbnails")
# DATA_PROXIES = os.path.join(DATA_ROOT, "proxies") # remove?

for d in (CLIPS_ROOT, DATA_THUMBS, DATA_EXPORTS):
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
    """create DB and table if not present."""
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
        exported_values TEXT,
        exported_filename TEXT
    ) WITHOUT ROWID;
    """)
    conn.commit()
    conn.close()

def init_scan():
    """Scan CLIPS_ROOT for new files, add critical UUID and filepath to DB and queue jobs."""
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
        # try:
        #     # Check if file is a valid video by attempting to probe it
        #     subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', os.path.join(CLIPS_ROOT, f)], stderr=subprocess.STDOUT)
        # except subprocess.CalledProcessError:
        #     # Not a valid video file

        new_uuid = str(uuid.uuid4())
        src_path = os.path.join(CLIPS_ROOT, f)
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute("""
        INSERT INTO clips (uuid, filename) VALUES (?, ?)
        """, (new_uuid, os.path.basename(src_path)))
        conn.commit()
        conn.close()
        jobsQueue.put(PriorityItem(10, new_uuid))  # Metadata
        # jobsQueue.put(PriorityItem(20, new_uuid))  # Proxy | TODO: copy data to fragmented mp4 container if its not
        jobsQueue.put(PriorityItem(30, new_uuid))  # Thumbnail
        print(f"🇬🇧 Discovered new clip: {f} as {new_uuid}")

def db_list_all_clips() -> List[dict]:
    """fetch all clips from DB, ordered by creation_time desc (most recent first)"""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT uuid, creation_time, size_bytes, duration, edited_title, filename, audio_tracks FROM clips ORDER BY creation_time DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

# --- Helpers ---
def ffprobe_duration(path: str) -> float:
    """ffprobe duration of a media file"""
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(path)}'
    out = subprocess.check_output(cmd, shell=True, text=True).strip()
    return float(out)

def compute_bitrates(size_limit_mb, duration, audio_kbps) -> int:
    """calc (calc is short for calculator) a bitrate audio/video pair within a size limit (MB)
    
    its the size limit in megabytes, converted into kilobits per second, without the inserted audio kilobits per second
    """
    size_kilobits = int(size_limit_mb * 8000)
    total_v_kbps = size_kilobits / max(0.1, duration)
    negated_video_kbps = int(total_v_kbps - audio_kbps)

    return negated_video_kbps


def ffprobe_trackcount(file_path: str) -> int: 
    """return how many audio tracks are in the file"""
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
                
                ext = "mp4" # configurable? should it be saved in the db or just when the job is created?
                edited_title = r[1] if r[1] else os.path.splitext(r[0])[0]
                # name will be: edited title, or filename without extension
                exported_file = os.path.join(DATA_EXPORTS, f"{sanitize(edited_title)}.{ext}")
                in_progress_file = os.path.join(DATA_EXPORTS, f"{export_uuid}.{ext}.clipthing")

                export_size_limit = item.export_limit_mb
                export_volumes = item.volumes
                export_start = item.start
                export_end = item.end

                # Bitrates
                duration = max(0.1, export_end - export_start)
                audio_kbps = 160 # hardcoded for now, TODO: UI choice

                export_video_kbps = compute_bitrates(export_size_limit, duration, audio_kbps)
                # format for ffmpeg args (strings like '500k')
                # export_video_bps_str = format_bitrate(export_video_bps)
                # export_audio_bps_str = format_bitrate(export_audio_bps)

                #filter complex logic
                filters = []
                map_inputs = []
                # ensure we have at least one volume level
                if not export_volumes:
                    export_volumes = [1.0]

                for index, vol in enumerate(export_volumes):
                    filters.append(f"[0:a:{index}]volume={vol}[a{index}]")
                    map_inputs.append(f"[a{index}]")
                amix = "".join(map_inputs) + f"amix=inputs={len(export_volumes)}:duration=longest[mixed]"
                audio_filter = "; ".join(filters + [amix])
                #end of the horror

                # Pass 1
                pass1 = [
                    "ffmpeg",
                    "-y", "-ss", str(export_start), "-t", str(duration),
                    "-i", input_file,
                    "-filter_complex", audio_filter,
                    "-map", "0:v:0", "-map", "[mixed]",
                    "-c:v", "libx264", "-preset", "medium", "-b:v", f"{export_video_kbps}k",
                    "-c:a", "aac", "-b:a", f"{audio_kbps}k",
                    "-pass", "1",
                    "-f", "mp4", os.devnull,
                ]
                p1 = subprocess.Popen(pass1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                out, two = p1.communicate()
                # print(out)

                    # "-movflags", "+faststart",
                print(f"{audio_filter}")
                # Pass 2
                pass2 = [
                    "ffmpeg",
                    "-y", "-ss", str(export_start), "-t", str(duration),
                    "-i", input_file,
                    "-filter_complex", audio_filter,
                    "-map", "0:v:0", "-map", "[mixed]",
                    "-c:v", "libx264", "-preset", "medium", "-b:v", f"{export_video_kbps}k",
                    "-c:a", "aac", "-b:a", f"{audio_kbps}k",
                    "-pass", "2", 
                    "-f", "mp4",
                    in_progress_file,
                ]
                print(pass2)
                    # "-movflags", "+faststart",
                p2 = subprocess.Popen(pass2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                out, two = p2.communicate()
                # print(out)
                os.rename(in_progress_file, exported_file)

                # store the exported filename in the DB
                conn = sqlite3.connect(DB_PATH)
                conn.execute("""
                            UPDATE clips SET exported_filename=? WHERE uuid=?""", 
                            (os.path.basename(exported_file), export_uuid))
                conn.commit()
                conn.close()


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

            # case 20: # Proxy TODO remake this entire case to remux into fragmented mp4 for native browser playback

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
                Warning("Unknown priority in job queue")
                jobsQueue.task_done()
                continue
        # except Exception as e:
        #     print(f"Error in worker loop: {e}")
        # finally:
        #     jobsQueue.task_done()

threading.Thread(target=worker_loop, name="worker", daemon=True).start()




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
        "creation_time": r["creation_time"],
        "audio_tracks": r["audio_tracks"]
    } for r in rows]

@app.get("/clips/{UUID}")
def get_clip(UUID: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    r = conn.execute("SELECT edited_volumes, edited_start, edited_stop, exported_values FROM clips WHERE uuid=?", (UUID,)).fetchone()
    conn.close()
    if not r: raise HTTPException(404, detail="clip not found")
    result = dict(r)
    # decode edited_volumes if it's a string
    if isinstance(result.get("edited_volumes"), str):
        try:
            result["edited_volumes"] = json.loads(result["edited_volumes"])
        except Exception:
            pass
    return result


class EditValues(BaseModel):
    edited_start: Optional[float] = None
    edited_stop: Optional[float] = None
    edited_volumes: Optional[List[float]] = None
    edited_title: Optional[str] = None

@app.patch("/clips/{UUID}")
def update_info(UUID: str, clip_data: EditValues):
    allowed_fields = {"edited_volumes", "edited_start", "edited_stop", "edited_title"}

    update_fields = {}
    for key, value in clip_data.model_dump().items():
        if key in allowed_fields and value is not None:
            update_fields[key] = value


    if not update_fields:
        raise HTTPException(400, detail=f"currently allowed editable fields are: {', '.join(allowed_fields)}")

    set_columns = ", ".join(f"{key}=?" for key in update_fields.keys())
    values = list(update_fields.values()) + [UUID]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"UPDATE clips SET {set_columns} WHERE uuid=?", values)
    conn.commit()
    rowcount = cur.rowcount
    conn.close()

    if not rowcount:
        raise HTTPException(404, detail="no changes made, does that UUID exist? or was the data identical?")

    return {"status": "updated"}

@app.delete("/clips/{UUID}")
def delete_clip(UUID: str):
    conn = sqlite3.connect(DB_PATH)
    r = conn.execute("SELECT filename FROM clips WHERE uuid=?", (UUID,)).fetchone()
    if not r:
        conn.close()
        raise HTTPException(404, detail="clip not found")
    filename = r[0]
    conn.execute("DELETE FROM clips WHERE uuid=?", (UUID,))
    conn.commit()
    conn.close()

    # delete the actual file
    file_path = os.path.join(CLIPS_ROOT, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    thumb_path = os.path.join(DATA_THUMBS, f"{UUID}.jpg")
    if os.path.exists(thumb_path):
        os.remove(thumb_path)


    return {"status": "deleted"}

@app.get("/clips/{UUID}/thumb")
def clip_thumb(UUID: str):
    conn = sqlite3.connect(DB_PATH)
    r = conn.execute("SELECT filename FROM clips WHERE uuid=?", (UUID,)).fetchone()
    conn.close()
    if not r: raise HTTPException(404)
    thumb_path = os.path.join(DATA_THUMBS, f"{UUID}.jpg")
    return FileResponse(thumb_path, media_type="image/jpeg")

@app.get("/clips/{UUID}/play")
def play(UUID: str):
    # proxy_path = os.path.join(DATA_PROXIES, f"{UUID}.mp4")
    conn = sqlite3.connect(DB_PATH)
    r = conn.execute("SELECT filename FROM clips WHERE uuid=?", (UUID,)).fetchone()
    conn.close()

    if not r:
        raise HTTPException(404, detail="No response from DB, does that clip exist?")
    filepath = os.path.join(CLIPS_ROOT, r[0])
    if not os.path.exists(filepath):
        raise HTTPException(404, detail="The item is tracked, but the file was not found on disk - has it been moved/renamed/deleted?")
    

    return FileResponse(filepath)

class QueueExportValues(BaseModel):
    start: float
    end: float
    volumes: List[float]
    size_limit_mb: float = 50.0 # hardcoded to 50mb TODO 

@app.post("/clips/{UUID}/export")
def queue_export(UUID: str, body: QueueExportValues):
    """queue an export job and store the export parameters in the DB"""
    exported_values = json.dumps(body.model_dump())

    # the body should contain the keys "start", "end", "volumes" (dict), "size_limit_mb"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""UPDATE clips SET
        exported_values=? 
    WHERE uuid=?
    """, (exported_values, UUID))
    conn.commit()
    rowcount = cur.rowcount
    conn.close()

    if not rowcount:
        # no rows updated -> clip not found
        raise HTTPException(404, detail="clip not found")

    jobsQueue.put(PriorityItem(
        1, UUID, body.start, body.end,
        body.volumes, body.size_limit_mb
    ))

    return {"status": "queued"}

@app.get("/clips/{UUID}/export")
def serve_export_file(UUID: str):

    conn = sqlite3.connect(DB_PATH)
    r = conn.execute("SELECT exported_filename FROM clips WHERE uuid=?", (UUID,)).fetchone()
    conn.close()

    if not r or not r[0]:
        raise HTTPException(404, detail="Prospective filename not found on database, has the export job been sent?")
    export_file = os.path.join(DATA_EXPORTS, r[0])

    if not os.path.exists(export_file):
        raise HTTPException(404, detail="Exported file not found on disk, has the export job completed or was the export folder cleared?")
    return FileResponse(export_file, filename=r[0])

@app.head("/clips/{UUID}/export")
def check_export_file(UUID: str):

    conn = sqlite3.connect(DB_PATH)
    r = conn.execute("SELECT exported_filename FROM clips WHERE uuid=?", (UUID,)).fetchone()
    conn.close()

    if not r or not r[0]:
        raise HTTPException(404, detail="Prospective filename not found on database, has the export job been sent?")
    export_file = os.path.join(DATA_EXPORTS, r[0])

    if not os.path.exists(export_file):
        raise HTTPException(404, detail="Exported file not found on disk, has the export job completed or was the export folder cleared?")
    return Response(status_code=200)


@app.get("/queue")
def queueSize():
    return {"queueSize": jobsQueue.qsize()}

def resource_path(relative_path: str) -> str:
    """ Get absolute path to resource, works for dev and PyInstaller """
    try:
        base_path = sys._MEIPASS  # pyright: ignore[reportAttributeAccessIssue] # Set by PyInstaller
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

WEB_ROOT = resource_path("")

@app.get("/", response_class=HTMLResponse)
def root():
    index = os.path.join(WEB_ROOT, "index.html")
    return FileResponse(index) if os.path.exists(index) else HTMLResponse("Missing index.html")

@app.get("/loading.jpg", response_class=HTMLResponse)
def loading():
    loading_path = os.path.join(WEB_ROOT, "loading.jpg")
    if os.path.exists(loading_path):
        return FileResponse(loading_path, media_type="image/jpeg")

    raise HTTPException(404)
    
# @app.get("/selectize.min.js", response_class=HTMLResponse)
# def selectizeJS():
#     index = os.path.join(WEB_ROOT, "selectize.min.js")
#     return FileResponse(index) if os.path.exists(index) else HTMLResponse("Missing selectize.min.js")

# @app.get("/selectize.bootstrap5.css", response_class=HTMLResponse)
# def selectizeCSS():
#     index = os.path.join(WEB_ROOT, "selectize.bootstrap5.css")
#     return FileResponse(index) if os.path.exists(index) else HTMLResponse("Missing selectize.bootstrap5.css")

# --- Startup ---
init_db()
init_scan()


if __name__ == "__main__":
    webbrowser.open("http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)