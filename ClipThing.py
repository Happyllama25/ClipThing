from __future__ import annotations

import json
import logging
import os
import queue
import shlex
import sqlite3
import subprocess
import sys
import threading
import uuid
import webbrowser

# from asyncio.queues import Queue
from dataclasses import dataclass, field
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from sanitize_filename import sanitize
from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

HOME_DIR = os.path.expanduser("~")
CLIPS_ROOT = os.environ.get(
    "CLIPS_ROOT", f"{HOME_DIR}/Videos/clips"
)  # on macos this works but it makes a new folder in the home dir bc mac calls it "Movies" cus they special

DATA_ROOT = os.path.join(CLIPS_ROOT, "data")
DATA_EXPORTS = os.path.join(CLIPS_ROOT, "exports")

# should we make a logs folder to contain ffmpeg logs? maybe futureproofing for multiple workers simultaneosly
DB_PATH = os.path.join(DATA_ROOT, "ClipThing.db")
DATA_THUMBS = os.path.join(DATA_ROOT, "thumbnails")
DATA_AUDIO = os.path.join(DATA_ROOT, "audio")
# DATA_PROXIES = os.path.join(DATA_ROOT, "proxies") # remove?

for d in (CLIPS_ROOT, DATA_THUMBS, DATA_EXPORTS):
    os.makedirs(d, exist_ok=True)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("ClipThing")
log.info("Logging Started!")

# --- Job definitions ---
jobsQueue = queue.PriorityQueue()
# notificationQueue = Queue()


@dataclass(order=True)
class PriorityItem:
    priority: int
    uuid: str = field(compare=False)
    start: Optional[float] = field(compare=False, default=None)
    end: Optional[float] = field(compare=False, default=None)
    # trim: Tuple[float, float] = field(compare=False)
    volumes: Optional[List[float]] = field(compare=False, default=None)
    export_limit_mb: Optional[float] = field(compare=False, default=None)


# --- Init DB ---
def init_db():
    """create DB and table if not present."""
    conn = get_db()
    conn.execute("PRAGMA journal_mode=WAL;")
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS clips (
        uuid TEXT PRIMARY KEY,
        process_status TEXT,
        user_status TEXT,
        publicity BOOLEAN,
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


def get_db():
    """get a new DB connection, with the right timeout and journal mode for concurrency"""
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


# def get_db_clip(uuid, columns):
#     conn = get_db()
#     cur = conn.cursor()
#     for col in columns:
# idk something like, get the clip and return the requested columns, maybe also retiring or combining the get_db function


def init_scan():
    """Scan CLIPS_ROOT for new files, add critical uuid and filepath to DB and queue jobs."""
    existing_files = set()
    conn = get_db()
    cur = conn.cursor()
    rows = cur.execute("SELECT filename FROM clips").fetchall()
    for r in rows:
        existing_files.add(r[0])
    conn.close()

    for file in os.listdir(CLIPS_ROOT):
        if file in existing_files:
            continue
        if not file.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv")):
            continue
        # try:
        #     # Check if file is a valid video by attempting to probe it
        #     subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', os.path.join(CLIPS_ROOT, f)], stderr=subprocess.STDOUT)
        # except subprocess.CalledProcessError:
        #     # Not a valid video file

        src_path = os.path.join(CLIPS_ROOT, file)
        ingest_new_clip(src_path)


def init_tray_icon():
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import QAction, QApplication, QMenu, QSystemTrayIcon

    def on_tray_activated(reason):
        if (
            reason == QSystemTrayIcon.activated
        ):  # dont think this even works lol, it was annoying anyway
            webbrowser.open("http://localhost:8000")

    qt_app = QApplication(sys.argv)

    if not QSystemTrayIcon.isSystemTrayAvailable():
        log.warning(
            "⚠️ System tray not available. Skipping icon. Consider adding the '-nogui' launch argument."
        )

    tray_icon = QSystemTrayIcon(QIcon(resource_path("icon.png")), qt_app)
    tray_icon.setToolTip("ClipThing")

    menu = QMenu()

    queue_status_action = QAction("Queue: fetching...")
    queue_status_action.setEnabled(False)
    menu.addAction(queue_status_action)

    menu.addSeparator()

    exit_action = QAction("Exit")
    exit_action.triggered.connect(qt_app.quit)
    menu.addAction(exit_action)

    tray_icon.setContextMenu(menu)
    tray_icon.activated.connect(on_tray_activated)
    tray_icon.show()

    def update_queue_status():
        job_count = jobsQueue.qsize()
        queue_status_action.setText(
            f"Queue: {job_count} job{'s' if job_count != 1 else ''}"
        )

    timer = QTimer()
    timer.timeout.connect(update_queue_status)
    timer.start(2000)  # every 2 seconds

    qt_app.exec_()


def db_list_all_clips() -> List[dict]:
    """fetch all clips from DB, ordered by creation_time desc (most recent first)"""
    conn = get_db()
    rows = conn.execute(
        "SELECT uuid, creation_time, size_bytes, duration, edited_title, filename, audio_tracks FROM clips ORDER BY creation_time DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Helpers ---


# async def notify(message: str, refresh: bool):
#     """notify UI of a change, and if a refresh is required (like if a clip was added or title change)"""
#     await notificationQueue.put({"message": message, "refresh": refresh})


def ffprobe_duration(path: str) -> float:
    """ffprobe duration of a media file"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def compute_bitrates(size_limit_mb, duration, audio_kbps) -> int:
    """calc (calc is short for calculator) a bitrate audio/video pair within a size limit (MB)

    its the size limit in megabytes, converted into kilobits per second, minus the inserted audio kilobits per second
    """
    size_kilobits = int(size_limit_mb * 8000)
    total_v_kbps = size_kilobits / max(0.1, duration)
    negated_video_kbps = int(total_v_kbps - audio_kbps)

    return negated_video_kbps


def ffprobe_trackcount(file_path: str) -> int:
    """return how many audio tracks are in the file"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        file_path,
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        log.error(f"ffprobe error for track count on {file_path}: {result.stderr}")
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    indexes = [stream["index"] for stream in data.get("streams", [])]
    return len(indexes)


def worker_loop():
    # jobsQueue.join()
    log.info("Worker thread started")
    while True:
        log.info("Worker waiting for job...")
        item = jobsQueue.get(block=True, timeout=None)
        log.info(f"Worker got job: {item} with priority {item.priority}")
        try:
            match item.priority:
                case 1:  # KILL KILL KILL 👹
                    jobsQueue.shutdown(
                        immediate=True
                    )  # make nicer - drain the queue into DB, save for next startup
                    break
                # case 2: # Pause
                #     log.info("Pausing worker thread")
                #     pause_event.clear()
                #     continue

                case 5:  # Export
                    # spawn ffmpeg, needed vars are file (deduce from uuid), start, end, volumes, size limit
                    export_uuid = item.uuid
                    # use DB to locate source video
                    conn = get_db()
                    r = conn.execute(
                        "SELECT filename, edited_title FROM clips WHERE uuid=?",
                        (export_uuid,),
                    ).fetchone()
                    conn.close()
                    if not r:
                        raise FileNotFoundError("clip not found")
                    input_file = os.path.join(CLIPS_ROOT, r[0])

                    ext = "mp4"  # configurable? should it be saved in the db or just when the job is created?
                    edited_title = r[1] if r[1] else os.path.splitext(r[0])[0]
                    # name will be: edited title, or filename without extension
                    exported_file = os.path.join(
                        DATA_EXPORTS, f"{sanitize(edited_title)}.{ext}"
                    )
                    in_progress_file = os.path.join(
                        DATA_EXPORTS, f"{export_uuid}.{ext}.clipthing"
                    )

                    export_size_limit = item.export_limit_mb
                    export_volumes = item.volumes
                    export_start = item.start
                    export_end = item.end

                    # Bitrates
                    duration = max(0.1, export_end - export_start)
                    audio_kbps = 160  # hardcoded for now, TODO: UI choice

                    export_video_kbps = compute_bitrates(
                        export_size_limit, duration, audio_kbps
                    )
                    # format for ffmpeg args (strings like '500k')
                    # export_video_bps_str = format_bitrate(export_video_bps)
                    # export_audio_bps_str = format_bitrate(export_audio_bps)

                    # filter complex logic
                    filters = []
                    map_inputs = []
                    # ensure we have at least one volume level
                    if not export_volumes:
                        export_volumes = [1.0]

                    for index, vol in enumerate(export_volumes):
                        filters.append(f"[0:a:{index}]volume={vol}[a{index}]")
                        map_inputs.append(f"[a{index}]")
                    amix = (
                        "".join(map_inputs)
                        + f"amix=inputs={len(export_volumes)}:duration=longest[mixed]"
                    )
                    audio_filter = "; ".join(filters + [amix])
                    # end of the horror

                    # Pass 1
                    pass1 = [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        str(export_start),
                        "-t",
                        str(duration),
                        "-i",
                        input_file,
                        "-filter_complex",
                        audio_filter,
                        "-map",
                        "0:v:0",
                        "-map",
                        "[mixed]",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        "-b:v",
                        f"{export_video_kbps}k",
                        "-c:a",
                        "aac",
                        "-b:a",
                        f"{audio_kbps}k",
                        "-pass",
                        "1",
                        "-f",
                        "mp4",
                        os.devnull,
                    ]
                    p1 = subprocess.Popen(
                        pass1,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    out, two = p1.communicate()
                    # log.debug(out)

                    # "-movflags", "+faststart",
                    log.debug(f"{audio_filter}")
                    # Pass 2
                    pass2 = [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        str(export_start),
                        "-t",
                        str(duration),
                        "-i",
                        input_file,
                        "-filter_complex",
                        audio_filter,
                        "-map",
                        "0:v:0",
                        "-map",
                        "[mixed]",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        "-b:v",
                        f"{export_video_kbps}k",
                        "-c:a",
                        "aac",
                        "-b:a",
                        f"{audio_kbps}k",
                        "-pass",
                        "2",
                        "-f",
                        "mp4",
                        in_progress_file,
                    ]
                    log.debug(f"{pass2}")
                    # "-movflags", "+faststart",
                    p2 = subprocess.Popen(
                        pass2,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    out, two = p2.communicate()
                    # log.debug(out)
                    os.rename(in_progress_file, exported_file)

                    # store the exported filename in the DB
                    conn = get_db()
                    conn.execute(
                        """
                                UPDATE clips SET exported_filename=? WHERE uuid=?""",
                        (os.path.basename(exported_file), export_uuid),
                    )
                    conn.commit()
                    conn.close()

                    continue

                case 10:  # Metadata
                    uuid = item.uuid
                    conn = get_db()
                    r = conn.execute(
                        "SELECT filename FROM clips WHERE uuid=?", (uuid,)
                    ).fetchone()
                    conn.close()
                    if not r:
                        log.error("file not found")
                        continue

                    filepath = os.path.join(CLIPS_ROOT, r[0])

                    creation_date = os.path.getmtime(filepath)
                    size_bytes = os.path.getsize(filepath)
                    try:
                        audio_tracks = ffprobe_trackcount(filepath)
                    except Exception:
                        audio_tracks = None

                    duration = ffprobe_duration(filepath)

                    conn = get_db()
                    conn.execute(
                        """
                    UPDATE clips SET
                        creation_time=?,
                        size_bytes=?,
                        audio_tracks=?,
                        duration=?
                    WHERE uuid=?
                    """,
                        (creation_date, size_bytes, audio_tracks, duration, uuid),
                    )
                    conn.commit()
                    conn.close()
                    continue

                case 15:  # Rip audio
                    uuid = item.uuid

                    conn = get_db()
                    r = conn.execute(
                        "SELECT filename, audio_tracks FROM clips WHERE uuid=?", (uuid,)
                    ).fetchone()
                    conn.close()
                    clip = dict(r)
                    if not clip:
                        log.error("file not found")
                        continue
                    clip_path = os.path.join(CLIPS_ROOT, clip["filename"])

                    ext = "m4a"  # TODO: configurable audio format
                    for track in range(clip["audio_tracks"]):
                        output_dir = os.path.join(DATA_AUDIO, uuid)
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(
                            output_dir, f"{uuid}_{track:02d}.{ext}"
                        )
                        cmd = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            clip_path,
                            "-map",
                            f"0:a:{track}",
                            "-c",
                            "copy",
                            output_file,
                        ]
                        result = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                        if (
                            result.returncode != 0
                        ):  # TODO: handle better, i think this kills the entire thread?
                            raise RuntimeError(
                                f"Failed to rip audio track {track} for {clip_path}: {result.stderr}"
                            )
                        continue

                # case 20: # Proxy TODO remake this entire case to remux into fragmented mp4 for better native browser playback if not already

                # uuid = item.uuid

                # conn = get_db()
                # r = conn.execute(
                #     "SELECT filename FROM clips WHERE uuid=?", (uuid,)
                # ).fetchone()
                # conn.close()
                # if not r:
                #     log.error("file not found")
                #     return

                # filepath = os.path.join(CLIPS_ROOT, r[0])

                # proxy_path = os.path.join(DATA_PROXIES, f"{uuid}.mp4")
                # cmd = (
                #     f"ffmpeg -y -i {shlex.quote(filepath)} -map 0:v:0 -map 0:a:0 "
                #     f"-c:v libx264 -preset veryfast -crf 23 -vf scale='min(1280\\,iw)':-2 "
                #     f"-c:a aac -b:a 128k -movflags +faststart {shlex.quote(proxy_path)}"
                # )
                # subprocess.call(cmd, shell=True)
                # continue

                case 30:  # Thumbnail
                    uuid = item.uuid

                    conn = get_db()
                    r = conn.execute(
                        "SELECT filename, duration FROM clips WHERE uuid=?", (uuid,)
                    ).fetchone()
                    conn.close()

                    if not r:
                        log.error("file not found")
                        return
                    filepath = os.path.join(CLIPS_ROOT, r[0])
                    thumb_path = os.path.join(DATA_THUMBS, f"{uuid}.jpg")

                    dur = (
                        r[1] if r[1] is not None else 2.0
                    )  ## ok that was faster and easier than expected

                    t = max(0.0, dur / 2.0)
                    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
                    thumbnail_ffmpeg = f'ffmpeg -y -ss {t:.2f} -i {shlex.quote(filepath)} -update true -frames:v 1 -vf "scale=min(480\\,iw):-2" {shlex.quote(thumb_path)}'

                    subprocess.check_call(thumbnail_ffmpeg, shell=True)

                    continue

                case _:
                    log.warning(f"Unknown priority in job queue {item.priority}")
                    continue
        except Exception as e:
            log.error(f"Error in worker thread for job {item}: {e}")
        finally:
            jobsQueue.task_done()


def start_worker_thread():  # TODO: fast and slow thread, one dedicated for exports - but also able to be cancelled (??)
    thread = threading.Thread(target=worker_loop, name="worker", daemon=True)
    thread.start()
    return thread


class ClipsDirHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.pending_timers = {}

    def on_modification(self, event: FileModifiedEvent):
        debounce_time = 5.0  # TODO configurable
        if event.is_directory:
            return
        if (
            str(event.src_path)
            .lower()
            .endswith((".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv"))
        ):
            # Cancel existing timer if any
            if event.src_path in self.pending_timers:
                self.pending_timers[event.src_path].cancel()
            # Start new timer to debounce
            self.pending_timers[event.src_path] = threading.Timer(
                debounce_time, self.process_file, args=(event.src_path,)
            )
            self.pending_timers[event.src_path].start()

    def process_file(self, path):
        self.pending_timers.pop(path, None)
        log.info(f"🌚 Watcher ingesting new clip at: {path}")
        ingest_new_clip(path)
        # notify("New Clip", 1)


def start_watchdog():
    observer = Observer()
    handler = ClipsDirHandler()
    observer.schedule(handler, CLIPS_ROOT, recursive=False)
    observer.start()
    log.info("Watcher thread started")
    return observer


def start_watcher_thread():
    observer = start_watchdog()

    def watcher_loop():
        observer.join()

    thread = threading.Thread(target=watcher_loop, name="watchdog", daemon=True)
    thread.start()

    return


def ingest_new_clip(file_path: str):
    new_uuid = str(uuid.uuid4())
    filename = os.path.basename(file_path)
    conn = get_db()
    conn.execute(
        """
    INSERT INTO clips (uuid, filename) VALUES (?, ?)
    """,
        (new_uuid, filename),
    )
    conn.commit()
    conn.close()
    jobsQueue.put(PriorityItem(10, new_uuid))  # Metadata
    jobsQueue.put(PriorityItem(15, new_uuid))  # Rip audio
    jobsQueue.put(PriorityItem(30, new_uuid))  # Thumbnail
    log.info(f"✨ Discovered new clip: {filename} as {new_uuid}")


# --- API ---
app = FastAPI(title="Clipper")


@app.get("/clips")
def list_clips():
    rows = db_list_all_clips()
    return [
        {
            "uuid": r["uuid"],
            "file": r["filename"],
            "size_bytes": r["size_bytes"],
            "edited_title": r["edited_title"],
            "creation_time": r["creation_time"],
            "audio_tracks": r["audio_tracks"],
        }
        for r in rows
    ]


@app.get("/clips/{uuid}")
def get_clip(uuid: str):
    conn = get_db()
    r = conn.execute(
        "SELECT edited_volumes, edited_start, edited_stop, exported_values FROM clips WHERE uuid=?",
        (uuid,),
    ).fetchone()
    conn.close()
    if not r:
        raise HTTPException(404, detail="clip not found")
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
    edited_audio_labels: Optional[List[str]] = None
    edited_title: Optional[str] = None


@app.patch("/clips/{uuid}")
def update_info(uuid: str, clip_data: EditValues):
    allowed_fields = {
        "edited_volumes",
        "edited_start",
        "edited_stop",
        "edited_title",
        "edited_audio_labels",
    }
    update_fields = {}
    for key, value in clip_data.model_dump().items():
        if key in allowed_fields and value is not None:
            update_fields[key] = value

    if not update_fields:
        raise HTTPException(
            400,
            detail=f"currently allowed editable fields are: {', '.join(allowed_fields)}",
        )

    set_columns = ", ".join(f"{key}=?" for key in update_fields.keys())
    values = list(update_fields.values()) + [uuid]

    conn = get_db()
    cur = conn.cursor()
    cur.execute(f"UPDATE clips SET {set_columns} WHERE uuid=?", values)
    conn.commit()
    rowcount = cur.rowcount
    conn.close()

    if not rowcount:
        raise HTTPException(
            404,
            detail="no changes made, does that uuid exist? or was the data identical?",
        )

    return {"status": "updated"}


@app.delete("/clips/{uuid}")
def delete_clip(uuid: str):
    conn = get_db()
    r = conn.execute("SELECT filename FROM clips WHERE uuid=?", (uuid,)).fetchone()
    if not r:
        conn.close()
        raise HTTPException(404, detail="clip not found")
    filename = r[0]
    conn.execute("DELETE FROM clips WHERE uuid=?", (uuid,))
    conn.commit()
    conn.close()

    # delete the actual file
    file_path = os.path.join(CLIPS_ROOT, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    thumb_path = os.path.join(DATA_THUMBS, f"{uuid}.jpg")
    if os.path.exists(thumb_path):
        os.remove(thumb_path)

    return {"status": "deleted"}


@app.get("/clips/{uuid}/thumb")
def clip_thumb(uuid: str):
    # conn = get_db() #why was all of this here? to verify if the uuid existed?
    # r = conn.execute("SELECT filename FROM clips WHERE uuid=?", (uuid,)).fetchone()
    # conn.close()
    # if not r: raise HTTPException(404)
    thumb_path = os.path.join(DATA_THUMBS, f"{uuid}.jpg")

    if not os.path.exists(thumb_path):
        jobsQueue.put(PriorityItem(30, uuid))
        raise HTTPException(
            status_code=404, detail="Thumbnail not found but one has been queued"
        )
    return FileResponse(thumb_path, media_type="image/jpeg")


@app.get("/clips/{uuid}/play")
def play(uuid: str):
    # proxy_path = os.path.join(DATA_PROXIES, f"{uuid}.mp4")
    conn = get_db()
    r = conn.execute("SELECT filename FROM clips WHERE uuid=?", (uuid,)).fetchone()
    conn.close()

    if not r:
        raise HTTPException(404, detail="No response from DB, does that clip exist?")
    filepath = os.path.join(CLIPS_ROOT, r[0])
    if not os.path.exists(filepath):
        raise HTTPException(
            404,
            detail="The item is tracked, but the file was not found on disk - has it been moved/renamed/deleted?",
        )

    return FileResponse(filepath)


@app.get("/clips/{uuid}/audio/{track_id}")
def get_audio_track(uuid: str, track_id: int):
    audio_path = os.path.join(DATA_AUDIO, uuid, f"{uuid}_{track_id:02d}.m4a")
    if not os.path.exists(audio_path):
        jobsQueue.put(PriorityItem(15, uuid))  # Rip audio
        raise HTTPException(
            202, detail="Audio tracks not found, ripping has been queued"
        )
    return FileResponse(audio_path, media_type="audio/m4a")


class QueueExportValues(BaseModel):
    start: float
    end: float
    volumes: List[float]
    size_limit_mb: float = 50.0  # hardcoded to 50mb TODO


@app.post("/clips/{uuid}/export")
def queue_export(uuid: str, body: QueueExportValues):
    """queue an export job and store the export parameters in the DB"""
    if body.start >= body.end:
        raise HTTPException(400, detail="start time must be less than end time")
    if body.size_limit_mb <= 0:
        raise HTTPException(400, detail="size limit must be greater than 0")

    exported_values = json.dumps(body.model_dump())

    # the body should contain the keys "start", "end", "volumes" (dict), "size_limit_mb"

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """UPDATE clips SET
        exported_values=?
    WHERE uuid=?
    """,
        (exported_values, uuid),
    )
    conn.commit()
    rowcount = cur.rowcount
    conn.close()

    if not rowcount:
        # no rows updated -> clip not found
        raise HTTPException(404, detail="clip not found")

    jobsQueue.put(
        PriorityItem(5, uuid, body.start, body.end, body.volumes, body.size_limit_mb)
    )

    return {"status": "queued"}


@app.get("/clips/{uuid}/export")
def serve_export_file(uuid: str):

    conn = get_db()
    r = conn.execute(
        "SELECT exported_filename FROM clips WHERE uuid=?", (uuid,)
    ).fetchone()
    conn.close()

    if not r or not r[0]:
        raise HTTPException(
            404,
            detail="Prospective filename not found on database, has the export job been sent?",
        )
    export_file = os.path.join(DATA_EXPORTS, r[0])

    if not os.path.exists(export_file):
        raise HTTPException(
            404,
            detail="Exported file not found on disk, has the export job completed or was the export folder cleared?",
        )
    return FileResponse(export_file, filename=r[0])


@app.head("/clips/{uuid}/export")
def check_export_file(uuid: str):

    conn = get_db()
    r = conn.execute(
        "SELECT exported_filename FROM clips WHERE uuid=?", (uuid,)
    ).fetchone()
    conn.close()

    if not r or not r[0]:
        raise HTTPException(
            404,
            detail="Prospective filename not found on database, has the export job been sent?",
        )
    export_file = os.path.join(DATA_EXPORTS, r[0])

    if not os.path.exists(export_file):
        raise HTTPException(
            404,
            detail="Exported file not found on disk, has the export job completed or was the export folder cleared?",
        )
    return Response(status_code=200)


@app.get("/queue")
def queueSize():
    return {"queueSize": jobsQueue.qsize()}


@app.post("/exit")
def exit():
    jobsQueue.put(PriorityItem(1, "EXIT-EXIT-EXIT-EXIT"))
    return {"queueSize": jobsQueue.qsize()}


def resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and PyInstaller"""
    try:
        base_path = sys._MEIPASS  # pyright: ignore[reportAttributeAccessIssue] # Set by PyInstaller
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


WEB_ROOT = resource_path("")


@app.get("/", response_class=HTMLResponse)
def root():
    index = os.path.join(WEB_ROOT, "index.html")
    return (
        FileResponse(index)
        if os.path.exists(index)
        else HTMLResponse("Missing index.html")
    )


@app.get("/loading.jpg", response_class=HTMLResponse)
def loading():
    loading_path = os.path.join(WEB_ROOT, "loading.jpg")
    if os.path.exists(loading_path):
        return FileResponse(loading_path, media_type="image/jpeg")

    raise HTTPException(404)


# --- Startup ---
if __name__ == "__main__":
    init_db()
    init_scan()

    start_worker_thread()
    start_watcher_thread()

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "-nogui":
                log.warning("⚠️   -nogui received, skipping tray icon...")
                uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        threading.Thread(
            target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000), daemon=True
        ).start()
        webbrowser.open("http://localhost:8000")
        init_tray_icon()
