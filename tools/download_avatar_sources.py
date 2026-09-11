"""Fetch nine CC0 Quaternius avatars from pinned public mirrors."""
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/avatar_grid/sources'
RPG = 'https://raw.githubusercontent.com/Akkuun/AreneDeMagiciens/13155fa18cb514b9c739fecf1614492de59840fc/jeu/arene-de-magicien/assets/Quaternius_RPG_Characters/RPG Characters - Nov 2020/'
WOMEN = 'https://raw.githubusercontent.com/hukasu/bevy-modular-characters/b84c33822779c827fec2074edbfe98c4ddafb0e5/assets/'


def fetch(url, path):
    url = quote(url, safe=':/')
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['curl', '-fLsS', '--retry', '2', url, '-o', str(path)], check=True)
    return {'url':url, 'path':str(path.relative_to(OUT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


manifest = []
for pack, names, remote, source_page in [
    ('rpg', ['Cleric','Monk','Ranger','Rogue','Warrior','Wizard'], RPG+'glTF/', 'https://quaternius.com/packs/rpgcharacters.html'),
    ('women', ['Adventurer','SciFi','Witch'], WOMEN, 'https://quaternius.com/packs/ultimatemodularwomen.html'),
]:
    for name in names:
        dest = OUT / pack / (name+'.gltf')
        files = [fetch(remote+name+'.gltf', dest)]
        gltf = json.loads(dest.read_text())
        for item in gltf.get('buffers',[]) + gltf.get('images',[]):
            uri = item.get('uri','')
            if uri and not uri.startswith('data:'):
                assert not uri.startswith('/') and '..' not in Path(uri).parts
                files.append(fetch(remote+uri, dest.parent/uri))
        manifest.append({'id':name.lower(), 'name':name, 'creator':'Quaternius', 'license':'CC0-1.0', 'source_page':source_page, 'model':str(dest.relative_to(OUT)), 'files':files})
        print('SOURCE_READY',name,flush=True)
fetch(RPG+'License.txt',OUT/'rpg/License.txt')
fetch('https://quaternius.com/packs/ultimatemodularwomen.html', OUT/'women/source_license.html')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
