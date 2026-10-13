"""Export timestamped recorded frames without resimulating an episode."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def export_video(episode,view="ego",output=None):
    episode=Path(episode).resolve()
    record=json.loads((episode/"episode.json").read_text())
    frames=record["frames"]
    if view not in record["views"] or len(frames)<2:
        raise ValueError("view is unavailable or fewer than two frames were recorded")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to export videos")
    output=Path(output) if output else episode/f"{view}.mp4"
    if output.exists():
        raise ValueError("video output already exists")
    output.parent.mkdir(parents=True,exist_ok=True)
    def quoted(frame):
        return str(episode/frame['images'][view]).replace("'","'\\''")
    lines=[]
    for first,second in zip(frames,frames[1:]):
        duration=second['time']-first['time']
        if duration>0:
            lines.extend([f"file '{quoted(first)}'",f"duration {duration:.9f}"])
    lines.append(f"file '{quoted(frames[-1])}'")
    with tempfile.NamedTemporaryFile(mode='w',suffix='.txt') as stream:
        stream.write('\n'.join(lines)+'\n');stream.flush()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','concat','-safe','0','-i',stream.name,
            '-t',str(frames[-1]['time']-frames[0]['time']),'-vf','fps=20','-c:v','libx264','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(output)],check=True)
    return {"video":str(output),"view":view,"duration_s":frames[-1]['time']-frames[0]['time']}
