"""Encode the 240-frame bust diagnostic and verify the delivered video."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/stitched_autorig')
p=parser.parse_args().root.resolve()
ffmpeg=shutil.which('ffmpeg');ffprobe=shutil.which('ffprobe');assert ffmpeg and ffprobe
assert len(list((p/'frames').glob('*.png')))==240
font=Path('/System/Library/Fonts/Supplemental/Arial.ttf')
filters=[]
if font.exists():
 for label,start,end in [('HEAD TURN / NOD',0,79/30),('NECK TILT / BEND',80/30,159/30),('ARM RAISE / ELBOW FLEX',160/30,239/30)]:
  filters.append(f"drawtext=fontfile='{font}':text='{label}':x=36:y=34:fontsize=24:fontcolor=white:enable='between(t,{start},{end})'")
args=[ffmpeg,'-y','-v','error','-framerate','30','-start_number','0','-i',str(p/'frames/%04d.png')]
if filters:args += ['-vf',','.join(filters)]
video=p/'evaluation.mp4'
subprocess.run(args+['-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
subprocess.run([ffmpeg,'-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
stream=json.loads(subprocess.check_output([ffprobe,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(video)],text=True))['streams'][0]
assert int(stream['nb_read_frames'])==240 and stream['r_frame_rate']=='30/1' and float(stream['duration'])==8.
report={'file':video.name,**stream,'decode_passed':True,'rigged_glb_sha256':hashlib.sha256((p/'rigged.glb').read_bytes()).hexdigest()}
(p/'render_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
