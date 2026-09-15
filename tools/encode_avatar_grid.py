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
# A preserved subset of cards provides a focused neckline comparison without
# embedding avatar identities into preparation or weighting rules.
def compare_cards(before,cards,filename):
    comparison = OUT/filename
    inputs, filters, layout, labels = [], [], [], []
    for row,(index,name) in enumerate(cards):
        for col,folder in enumerate((before,OUT)):
            stream = row*2+col
            inputs += ['-i',folder/f'{name}_demo.mp4']
            filters.append(f'[{stream}:v]crop=340:340:122:40[v{stream}]')
            labels.append(f'[v{stream}]')
            layout.append(f'{col*340}_{row*340}')
    filters.append(''.join(labels)+f'xstack=inputs={len(labels)}:layout='+ '|'.join(layout)+
                   f',pad=680:{len(cards)*340+64}:0:64:color=0x101c28,'+
                   "drawtext=text='BEFORE':fontcolor=white:fontsize=24:x=120:y=20,"+
                   "drawtext=text='AFTER':fontcolor=white:fontsize=24:x=465:y=20"+
                   ''.join(f",drawtext=text='{index+1:02d}':fontcolor=white:fontsize=22:x=12:y={row*340+76}"
                           for row,(index,_) in enumerate(cards)))
    ffmpeg(*inputs,'-filter_complex',';'.join(filters),'-c:v','libx264','-crf','18',
           '-pix_fmt','yuv420p','-movflags','+faststart',comparison)
    videos.append((comparison,720))

before = OUT/'before_neck_cloth_fix'
entries = json.loads((OUT/'sources/manifest.json').read_text())
cards = [(i,e['id']) for i,e in enumerate(entries) if (before/(e['id']+'_demo.mp4')).exists()]
if cards:
    compare_cards(before,cards,'neck_cloth_before_after.mp4')
before = OUT/'before_voxel_seam_fix'
audit = OUT/'seam_validation_before.json'
if audit.exists():
    worst = sorted(json.loads(audit.read_text())['avatars'],
                   key=lambda c:c['max_gap_growth_360_frames'],reverse=True)[:2]
    names = {c['id'] for c in worst}
    cards = [(i,e['id']) for i,e in enumerate(entries)
             if e['id'] in names and (before/(e['id']+'_demo.mp4')).exists()]
    if cards:
        compare_cards(before,cards,'voxel_seams_before_after.mp4')
metadata = []
for video,expected in videos:
    result = subprocess.check_output([FFPROBE,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(video)],text=True)
    stream = json.loads(result)['streams'][0]
    assert int(stream['nb_read_frames']) == expected
    assert stream['r_frame_rate'] == '30/1'
    assert abs(float(stream['duration'])-expected/30)<.01
    ffmpeg('-xerror','-i',video,'-f','null','-')
    metadata.append({'file':video.name,**stream,'decode_passed':True})
(OUT/'render_validation.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2))
