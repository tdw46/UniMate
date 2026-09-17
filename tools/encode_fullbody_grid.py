"""Encode and decode-check both full-body 3x3 evaluation views."""
import hashlib,json,shutil,subprocess
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'outputs/fullbody_avatar_grid'
ffmpeg=shutil.which('ffmpeg');ffprobe=shutil.which('ffprobe');assert ffmpeg and ffprobe
report=[]
for view in ('overview','tour'):
 frames=out/'frames'/view
 assert len(list(frames.glob('*.png')))==720
 video=out/f'fullbody_{view}.mp4'
 subprocess.run([ffmpeg,'-y','-v','error','-framerate','30','-start_number','0','-i',str(frames/'%04d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
 subprocess.run([ffmpeg,'-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
 stream=json.loads(subprocess.check_output([ffprobe,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(video)],text=True))['streams'][0]
 assert stream['width']==1920 and stream['height']==1920
 assert int(stream['nb_read_frames'])==720 and stream['r_frame_rate']=='30/1' and float(stream['duration'])==24.
 report.append({'view':view,**stream,'decode_passed':True,'sha256':hashlib.sha256(video.read_bytes()).hexdigest()})
(out/'render_validation.json').write_text(json.dumps({'videos':report,'passed':True},indent=2));print(json.dumps(report,indent=2))
