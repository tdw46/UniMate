"""Encode and decode-verify the nine-avatar grid and individual card demos."""
import json
import os
from pathlib import Path
import shutil
import subprocess

OUT = Path(os.environ.get('AVATAR_EVAL_ROOT', Path(__file__).resolve().parents[1]/'outputs/avatar_grid')).resolve()
FFMPEG,FFPROBE = shutil.which('ffmpeg'),shutil.which('ffprobe')
assert FFMPEG and FFPROBE


def ffmpeg(*args):
    subprocess.run([FFMPEG,'-y','-v','error',*map(str,args)],check=True)


videos = []
for view in ('front','oblique'):
    folder = OUT/'frames'/view
    assert len(list(folder.glob('*.png'))) == 360, folder
    video = OUT/f'avatar_grid_{view}.mp4'
    ffmpeg('-framerate','30','-start_number','0','-i',folder/'%04d.png','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',video)
    videos.append((video,360))
concat = OUT/'concat.txt'
concat.write_text("file 'avatar_grid_front.mp4'\nfile 'avatar_grid_oblique.mp4'\n")
movie = OUT/'nine_avatar_grid.mp4'
ffmpeg('-f','concat','-safe','0','-i',concat,'-c','copy','-movflags','+faststart',movie)
videos.append((movie,720))
for index,entry in enumerate(json.loads((OUT/'sources/manifest.json').read_text())):
    col,row = index%3,index//3
    video = OUT/(entry['id']+'_demo.mp4')
    x = [64,668,1270][col]
    y = [170,706,1242][row]
    ffmpeg('-i',movie,'-vf',f'crop=584:496:{x}:{y}','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',video)
    videos.append((video,720))
ffmpeg('-i',movie,'-vf',"select='eq(n,35)'",'-frames:v','1',OUT/'nine_avatar_grid_poster.jpg')
ffmpeg('-i',movie,'-vf',"select='eq(n,35)+eq(n,155)+eq(n,299)+eq(n,395)+eq(n,515)+eq(n,659)',scale=640:640,tile=3x2",'-frames:v','1',OUT/'contact_sheet.jpg')
# When an earlier evaluation is present, retain a close-up regression comparison.
before = OUT/'before_apparel_fix'
if all((before/f'{name}_demo.mp4').exists() for name in ('monk','scifi')):
    comparison = OUT/'apparel_before_after.mp4'
    inputs = []
    for name in ('monk','scifi'):
        for folder in (before, OUT):
            inputs += ['-i',folder/f'{name}_demo.mp4']
    ffmpeg(*inputs,'-filter_complex',
           '[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|584_0|0_496|584_496,'
           'pad=1168:1056:0:64:color=0x101c28,'
           "drawtext=text='BEFORE':fontcolor=white:fontsize=28:x=225:y=18,"
           "drawtext=text='CORRECTED':fontcolor=white:fontsize=28:x=795:y=18",
           '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',comparison)
    videos.append((comparison,720))
metadata = []
for video,expected in videos:
    result = subprocess.check_output([FFPROBE,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(video)],text=True)
    stream = json.loads(result)['streams'][0]
    assert int(stream['nb_read_frames']) == expected
    assert stream['r_frame_rate'] == '30/1'
    assert abs(float(stream['duration'])-expected/30)<.01
    metadata.append({'file':video.name,**stream})
(OUT/'render_validation.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2))
