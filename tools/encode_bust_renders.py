"""Encode render sequences and verify their duration, rate and frame count."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/bust_evaluation'
ffmpeg = shutil.which('ffmpeg')
ffprobe = shutil.which('ffprobe')
assert ffmpeg and ffprobe, 'ffmpeg and ffprobe are required'
videos = []
for clip in ('head','neck','arms'):
    for view in ('front','oblique'):
        folder = OUT / 'frames' / f'{clip}_{view}'
        assert len(list(folder.glob('*.png'))) == 120, folder
        video = OUT / f'{clip}_{view}.mp4'
        subprocess.run([ffmpeg,'-y','-loglevel','error','-framerate','30','-start_number','0','-i',str(folder / '%04d.png'),'-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)], check=True)
        videos.append(video)
concat = OUT / 'video_concat.txt'
concat.write_text(''.join(f"file '{v.name}'\n" for v in videos))
movie = OUT / 'unimate_bust_evaluation.mp4'
subprocess.run([ffmpeg,'-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(movie)], check=True)
metadata = []
for video in videos + [movie]:
    result = subprocess.check_output([ffprobe,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(video)], text=True)
    stream = json.loads(result)['streams'][0]
    expected = 720 if video == movie else 120
    assert int(stream['nb_read_frames']) == expected
    assert stream['r_frame_rate'] == '30/1'
    assert abs(float(stream['duration']) - expected/30) < .01
    metadata.append({'file':video.name, **stream})
(OUT / 'render_validation.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))
