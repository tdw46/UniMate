"""Download the anime evaluation sources with pinned URLs and creator licenses."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/complex_avatar_grid'
COMMIT = 'e16eb187100149a315ad92c3c9968f1d5baa6c7d'
SOURCES = [
    ('Darkness_Shibu','beta','360012381793'),
    ('Sakurada_Fumiriya','beta','360014788554'),
    ('Sendagaya_Shino','beta','360013482714'),
    ('Victoria_Rubin','beta','360014900233'),
    ('Vita','beta','360014900113'),
    ('Vivi','beta','360014900273'),
    ('AvatarSample_A','stable','4402394424089'),
    ('AvatarSample_B','stable','4402394424089'),
    ('AvatarSample_C','stable','4402394424089'),
]

manifest = []
folder = OUT/'sources'
folder.mkdir(parents=True, exist_ok=True)
for name, version, article in SOURCES:
    url = f'https://raw.githubusercontent.com/madjin/vrm-samples/{COMMIT}/vroid/{version}/{name}.vrm'
    target = folder/(name+'.vrm')
    subprocess.run(['curl','-fLsS','--retry','2',url,'-o',str(target)],check=True)
    data = target.read_bytes()
    length = struct.unpack_from('<I',data,12)[0]
    gltf = json.loads(data[20:20+length])
    meta = gltf['extensions']['VRM']['meta']
    page = 'https://vroid.pixiv.help/hc/en-us/articles/'+article
    license_path = folder/('license_'+article+'.html')
    license_snapshot = license_path.exists()
    if not license_snapshot:
        # The creator's help center may block command-line downloads. Keep the
        # primary license URL and embedded VRM terms even when HTML is blocked.
        result = subprocess.run(['curl','-fLsS','--retry','2','-A','Mozilla/5.0',page,'-o',str(license_path)])
        license_snapshot = result.returncode == 0
    label = name.replace('_',' ')
    manifest.append({'id':name.lower(), 'name':label,
                     'display_name':label.replace('AvatarSample','Sample').replace('Sakurada ','').replace('Sendagaya ',''),
                     'creator':'pixiv / VRoid Project',
                     'license':'CC0-1.0' if version=='beta' else 'VRoidPreset terms (not CC0)',
                     'source_page':page, 'license_html_saved':license_snapshot,
                     'model':target.name, 'vrm_meta':meta,
                     'files':[{'url':url,'path':target.name,'sha256':hashlib.sha256(data).hexdigest()}]})
    print('SOURCE_READY',name,flush=True)
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
(OUT/'dataset.json').write_text(json.dumps({
    'title':'ANIME OUTFIT STUDY / 09',
    'renderer':'EEVEE',
    'credits':'VRoid / pixiv sources  |  Applied Boolean busts  |  New rigs + new weights',
},indent=2))
