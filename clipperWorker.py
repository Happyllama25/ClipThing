from __future__ import annotations
import os, shlex, json, time, uuid, threading, queue, signal, subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel


IMPORTS = os.environ.get("IMPORTS", "./imports")
CLIPS   = os.environ.get("CLIPS", "./clips")
EXPORTS = os.environ.get("EXPORTS", "./exports")
PROXY   = os.environ.get("PROXY", "./proxy")
THUMBS  = os.environ.get("THUMBS", "./thumbs")

os.makedirs(IMPORTS, exist_ok=True)
os.makedirs(CLIPS, exist_ok=True)
os.makedirs(EXPORTS, exist_ok=True)
os.makedirs(PROXY, exist_ok=True)
os.makedirs(THUMBS, exist_ok=True)

@dataclass
class Job:
    id: str
    clip_id: str
    start: float
    end: float
    volumes: list[float]
    size_limit_mb: float
    audio_kbps: int
    two_pass: bool
    created_at: float = field(default_factory=time.time)
    status: str = "queued"  # queued, running, done, failed
    progress: float = 0.0   # 0.0 to 1.0
    message: str = ""
    result_path: Optional[str] = None
    video_bps: Optional[int] = None
    audio_bps: Optional[int] = None
    duration: Optional[float] = None
    logs: List[str] = field(default_factory=list)
    _popen: Optional[subprocess.Popen] = field(default=None, repr=False, compare=False)
    _cancel: bool = field(default=False, repr=False, compare=False)


# --- In-memory registry & queue ---
JOBS: Dict[str, Job] = {}
Q: "queue.Queue[str]" = queue.Queue()
LOCK = threading.RLock()

def ffprobe_duration(path: str) -> float:
    """Get duration of media file using ffprobe."""
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(path)}'
    out = subprocess.check_output(cmd, shell=True, text=True).strip()
    return float(out)

def compute_bitrates(size_limit_mb: float, duration: float, audio_kbps: int) -> tuple[int, int]:
    """
    Calculate video and audio bitrates to fit a size limit for a given duration.

    Args:
        size_limit_mb (float): The maximum allowed file size in megabytes.
        duration (float): The duration of the media in seconds.
        audio_kbps (int): Desired audio bitrate in kilobits per second.

    Returns:
        tuple[int, int]: A tuple containing (video_bps, audio_bps) in bits per second.
    """
    size_bytes = int(size_limit_mb * 1024 * 1024)
    if duration <= 0:
        raise ValueError("Duration must be greater than zero for bitrate calculation.")
    total_bps = (size_bytes * 8) / duration
    audio_bps = max(64_000, int(audio_kbps * 1000))
    video_bps = int(max(100_000, total_bps - audio_bps))
    return video_bps, audio_bps

def parse_progress_line(line: str) -> dict:
    kv = {}
    line = line.strip()
    if not line or "=" not in line:
        return kv
    k, v = line.split("=", 1)
    kv[k] = v
    return kv

# Core ffmpeg runner (two-pass optional) with progress updates

def run_ffmpeg_export(job: Job) -> None:
    src = os.path.join(CLIPS, job.clip_id + ".mkv")
    if not os.path.exists(src):
        raise FileNotFoundError("Clip not found: " + src)

    # duration of trimmed segment
    duration = max(0.1, job.end - job.start)
    job.duration = duration
    video_bps, audio_bps = compute_bitrates(job.size_limit_mb, duration, job.audio_kbps)
    job.video_bps, job.audio_bps = video_bps, audio_bps

    # Build volume/mix filter for up to 4 tracks
    vols = (job.volumes + [0, 0, 0, 0])[:4]
    labels = []
    vol_parts = []
    for i, v in enumerate(vols):
        if v <= 0.0001:
            continue
        vol_parts.append(f"[0:a:{i}]volume={max(0.0, v)}[a{i}]")
        labels.append(f"[a{i}]")
    if not labels:
        # if all muted, feed silence from a nullsrc audio
        mix = "anullsrc=r=48000:cl=stereo,atrim=duration=" + str(duration) + "[outa]"
        filter_complex = mix
        map_audio = "-map [outa]"
    else:
        mix = "".join(labels) + f"amix=inputs={len(labels)}:normalize=0,aresample=48000[outa]"
        prefix = ";".join(vol_parts)
        filter_complex = prefix + ";" + mix if prefix else mix
        map_audio = "-map [outa]"

    base_out = f"{job.clip_id}_{int(time.time())}_export.mp4"
    out_path = os.path.join(EXPORTS, base_out)

    def spawn(cmd: str) -> subprocess.Popen:
        # use text mode to parse -progress lines
        return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # helper to read progress
    def pump_progress(p: subprocess.Popen, pass_idx: int, total_passes: int):
        last_ms = 0
        while True:
            if job._cancel:
                try:
                    p.send_signal(signal.SIGINT)
                except Exception:
                    pass
            line = p.stdout.readline()  # type: ignore
            if not line:
                if p.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            job.logs.append(line.rstrip())
            kv = parse_progress_line(line)
            if "out_time_ms" in kv:
                try:
                    out_ms = int(kv["out_time_ms"])  # cumulative encoded time in ms
                    last_ms = out_ms
                    frac = min(1.0, max(0.0, (out_ms / 1000.0) / duration))
                    # blend passes: e.g., pass 1 = 0..0.45, pass 2 = 0.55..1.0
                    if total_passes == 2:
                        base = 0.45 if pass_idx == 1 else 0.0
                        scale = 0.45 if pass_idx == 0 else 0.55
                        job.progress = min(0.99, base + frac * scale)
                    else:
                        job.progress = min(0.99, frac)
                except Exception:
                    pass
        # finalize per-pass
        if p.returncode == 0:
            if total_passes == 2 and pass_idx == 0:
                job.progress = max(job.progress, 0.45)
            else:
                job.progress = max(job.progress, 0.99)

    # Build commands
    common_trim = f"-ss {job.start} -to {job.end} -i {shlex.quote(src)}"

    if job.two_pass:
        # PASS 1 (no audio)
        cmd1 = (
            f"ffmpeg -y {common_trim} -map 0:v:0 -c:v libx264 -preset veryfast "
            f"-b:v {video_bps} -pass 1 -an -progress pipe:1 -f mp4 NUL"
        )
        # PASS 2 (with audio)
        cmd2 = (
            f"ffmpeg -y {common_trim} -filter_complex \"{filter_complex}\" -map 0:v:0 {map_audio} "
            f"-c:v libx264 -preset veryfast -b:v {video_bps} -maxrate {video_bps} -bufsize {2*video_bps} "
            f"-c:a aac -b:a {audio_bps} -pass 2 -movflags +faststart -progress pipe:1 {shlex.quote(out_path)}"
        )
        # run pass 1
        p1 = spawn(cmd1); job._popen = p1; pump_progress(p1, pass_idx=0, total_passes=2)
        if p1.wait() != 0:
            raise RuntimeError("ffmpeg pass 1 failed")
        # run pass 2
        p2 = spawn(cmd2); job._popen = p2; pump_progress(p2, pass_idx=1, total_passes=2)
        if p2.wait() != 0:
            raise RuntimeError("ffmpeg pass 2 failed")
    else:
        cmd = (
            f"ffmpeg -y {common_trim} -filter_complex \"{filter_complex}\" -map 0:v:0 {map_audio} "
            f"-c:v libx264 -preset veryfast -b:v {video_bps} -maxrate {video_bps} -bufsize {2*video_bps} "
            f"-c:a aac -b:a {audio_bps} -movflags +faststart -progress pipe:1 {shlex.quote(out_path)}"
        )
        p = spawn(cmd); job._popen = p; pump_progress(p, pass_idx=0, total_passes=1)
        if p.wait() != 0:
            raise RuntimeError("ffmpeg failed")

    job.result_path = out_path

# Worker loop

def worker_loop():
    while True:
        job_id = Q.get()  # blocking
        with LOCK:
            job = JOBS.get(job_id)
        if not job:
            continue
        if job.status == "canceled":
            continue
        try:
            job.status = "running"; job.message = "encoding"
            run_ffmpeg_export(job)
            if job._cancel:
                job.status = "canceled"; job.message = "canceled by user"
            else:
                job.progress = 1.0
                job.status = "done"; job.message = "complete"
        except Exception as e:
            job.status = "failed"; job.message = str(e)
        finally:
            # clear process handle
            job._popen = None
            Q.task_done()

# Start worker thread
threading.Thread(target=worker_loop, daemon=True).start()

# --- FastAPI app & schemas ---
app = FastAPI(title="Encode Queue")

class ExportReq(BaseModel):
    start: float
    end: float
    volumes: List[float] = [1.0, 1.0, 1.0, 1.0]
    size_limit_mb: float = 50.0
    audio_kbps: int = 160
    two_pass: bool = True

@app.post("/clips/{clip_id}/export")
def enqueue_export(clip_id: str, req: ExportReq):
    src = os.path.join(CLIPS, clip_id + ".mkv")
    if not os.path.exists(src):
        raise HTTPException(404, "Clip not found")
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        clip_id=clip_id,
        start=req.start,
        end=req.end,
        volumes=req.volumes,
        size_limit_mb=req.size_limit_mb,
        audio_kbps=req.audio_kbps,
        two_pass=req.two_pass,
    )
    with LOCK:
        JOBS[job_id] = job
    Q.put(job_id)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": job.status})

@app.get("/jobs")
def list_jobs():
    with LOCK:
        items = [asdict(j) for j in sorted(JOBS.values(), key=lambda x: x.created_at, reverse=True)[:100]]
    # remove private fields before returning
    for it in items:
        it.pop("_popen", None); it.pop("_cancel", None)
    return items

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    data = asdict(job)
    data.pop("_popen", None); data.pop("_cancel", None)
    return data

@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        if job.status in ("done", "failed", "canceled"):
            return {"ok": False, "message": f"cannot cancel status={job.status}"}
        job._cancel = True
        job.status = "canceled"
        job.message = "cancel requested"
        if job._popen and job._popen.poll() is None:
            try:
                job._popen.send_signal(signal.SIGINT)
            except Exception:
                pass
    return {"ok": True}

@app.get("/exports/{name}")
def serve_export(name: str):
    path = os.path.join(EXPORTS, name)
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4", filename=name)

# Healthcheck
@app.get("/health")
def health():
    return {"ok": True}

# --- Run ---
# uvicorn clipperWorker:app --reload --port 8000

# -------------------------------
# Clip listing + thumbnails + proxy playback
# -------------------------------

def _clip_path(clip_id: str) -> str:
    return os.path.join(CLIPS, f"{clip_id}.mkv")

def _proxy_path(clip_id: str) -> str:
    return os.path.join(PROXY, f"{clip_id}.proxy.mp4")

def _thumb_path(clip_id: str) -> str:
    return os.path.join(THUMBS, f"{clip_id}.jpg")

def ffprobe_json(path: str) -> dict:
    cmd = f'ffprobe -v error -show_streams -show_format -of json {shlex.quote(path)}'
    out = subprocess.check_output(cmd, shell=True, text=True)
    return json.loads(out)

def make_thumbnail_sync(src: str, dst: str):
    try:
        dur = ffprobe_duration(src)
    except Exception:
        dur = 2.0
    t = max(0.0, dur / 2.0)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = (
        f'ffmpeg -y -ss {t:.2f} -i {shlex.quote(src)} -frames:v 1 '
        f'-vf "scale=min(480\\,iw):-2" {shlex.quote(dst)}'
    )
    subprocess.check_call(cmd, shell=True)

@app.get("/clips")
def list_clips():
    # Discover mkv files and return metadata suitable for a grid view
    items = []
    for name in sorted(os.listdir(CLIPS)):
        if not name.lower().endswith(".mkv"):
            continue
        clip_id = os.path.splitext(name)[0]
        src = _clip_path(clip_id)
        try:
            meta = ffprobe_json(src)
            duration = float(meta.get("format", {}).get("duration", 0))
            a_tracks = len([s for s in meta.get("streams", []) if s.get("codec_type") == "audio"])
        except Exception:
            duration, a_tracks = 0.0, 0
        stat = os.stat(src)
        thumb = _thumb_path(clip_id)
        has_thumb = os.path.exists(thumb)
        proxy = _proxy_path(clip_id)
        has_proxy = os.path.exists(proxy)
        items.append({
            "id": clip_id,
            "file": name,
            "size_bytes": stat.st_size,
            "modified_ts": int(stat.st_mtime),
            "duration": duration,
            "audio_tracks": a_tracks,
            "thumb_url": f"/clips/{clip_id}/thumb",
            "play_url": f"/clips/{clip_id}/play",
            "has_thumb": has_thumb,
            "has_proxy": has_proxy,
        })
    return items

@app.get("/clips/{clip_id}/thumb")
def clip_thumb(clip_id: str):
    src = _clip_path(clip_id)
    if not os.path.exists(src):
        raise HTTPException(404, "Clip not found")
    dst = _thumb_path(clip_id)
    if not os.path.exists(dst):
        try:
            make_thumbnail_sync(src, dst)
        except subprocess.CalledProcessError:
            # fallback: serve missing.jpg
            return FileResponse("./Missingno.jpg", media_type="image/jpeg")
    return FileResponse(dst, media_type="image/jpeg")

@app.get("/clips/{clip_id}/play")
def play_proxy(clip_id: str):
    dst = _proxy_path(clip_id)
    if not os.path.exists(dst):
        raise HTTPException(404, "Proxy not ready. POST /clips/{clip_id}/prepare_proxy then retry.")
    return FileResponse(dst, media_type="video/mp4", filename=os.path.basename(dst))

@app.post("/clips/{clip_id}/prepare_proxy")
def prepare_proxy(clip_id: str):
    src = _clip_path(clip_id)
    if not os.path.exists(src):
        raise HTTPException(404, "Clip not found")
    dst = _proxy_path(clip_id)
    if os.path.exists(dst):
        return {"status": "ready", "play_url": f"/clips/{clip_id}/play"}

    def _make():
        # quick 720p preview proxy
        cmd = (
            f"ffmpeg -y -i {shlex.quote(src)} -map 0:v:0 -map 0:a:0 "
            f"-c:v libx264 -preset veryfast -crf 23 -vf scale='min(1280\\,iw)':-2 "
            f"-c:a aac -b:a 128k -movflags +faststart {shlex.quote(dst)}"
        )
        subprocess.call(cmd, shell=True)
    threading.Thread(target=_make, daemon=True).start()
    return JSONResponse(status_code=202, content={"status": "preparing"})