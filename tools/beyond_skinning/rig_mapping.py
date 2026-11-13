"""Independent, bounded name/hierarchy mapping; not full geometric rig_map.

Finger regexes recovered from the pinned binary's twelve-family constructor
0x155d68. No original binary or recorded oracle output is read at runtime.
"""
import re

def normalize(name):
    return name.split(':', 1)[-1].strip().lower()

def validate(names, parents, deform):
    n=len(names)
    if len(parents)!=n:
        raise ValueError('invalid research skeleton size')
    p=[int(x) for x in parents]
    if any(x < -1 or x >= n for x in p):raise ValueError('invalid parent index')
    # Reuse verified ancestry on long chains. A visiting node denotes a
    # cycle only during the current walk; previous walks finish as verified.
    state=bytearray(n)
    for start in range(n):
        i=start
        while i>=0 and state[i]==0:
            state[i]=1
            i=p[i]
        if i>=0 and state[i]==1:
            raise ValueError('cyclic skeleton')
        i=start
        while i>=0 and state[i]==1:
            state[i]=2
            i=p[i]
    d=[True]*n if deform is None else list(map(bool,deform))
    if len(d)!=n:raise ValueError('invalid deform mask')
    return list(map(normalize,names)),p,d

FINGER_PATTERNS = [('mixamo', '^(?P<side>left|right)hand(?P<digit>thumb|index|middle|ring|pinky|little|mid|f_index|f_middle|f_ring|f_pinky)(?P<n>\\d)$'), ('arp', '^(?:c_)?(?P<digit>thumb|index|middle|ring|pinky|little|mid|f_index|f_middle|f_ring|f_pinky)(?P<n>\\d)(?P<meta>_base)?(?:_bend)?\\.(?P<side>l|r)$'), ('rigify', '^(?:def-)?(?:(?P<digit>thumb|index|middle|ring|pinky|little|mid|f_index|f_middle|f_ring|f_pinky)\\.(?P<n>\\d{2})|palm_(?P<palm>index|middle|ring|pinky)|palm\\.(?P<palmno>0[1-4]))\\.(?P<side>l|r)$'), ('daz', '^(?P<side>l|r)_?(?P<digit>thumb|index|mid|ring|pinky|bigtoe|indextoe|midtoe|ringtoe|pinkytoe|smalltoe\\d)(?:(?P<meta>metacarpal)|_?(?P<n>\\d))?$'), ('mmd', '^(?P<side>左|右)(?P<digit>親指|人指|中指|薬指|小指)(?P<n>[0-3０-３])$'), ('mmd_en', '^(?P<digit>thumb|fore|middle|third|little)(?P<n>\\d)_(?P<side>l|r)$'), ('biped', '^bip\\d+[ _-](?P<side>l|r)[ _-](?P<digit>finger\\d|toe\\d)(?P<n>\\d)?$'), ('unreal', '^(?P<digit>thumb|index|middle|ring|pinky|little|mid|f_index|f_middle|f_ring|f_pinky)_(?:(?P<n>\\d{2})_|(?P<meta>metacarpal)_|(?P<tip>tip)_)?(?P<side>l|r)$'), ('cc', '^cc_base_(?P<side>l|r)_(?P<digit>thumb|index|mid|ring|pinky|bigtoe|indextoe|midtoe|ringtoe|pinkytoe)(?P<n>\\d)$'), ('vrm', '^j_bip_(?P<side>l|r)_(?P<digit>thumb|index|middle|ring|pinky|little|mid|f_index|f_middle|f_ring|f_pinky)(?P<n>\\d)$'), ('secondlife', '^mhand(?P<digit>thumb|index|middle|ring|pinky|little|mid|f_index|f_middle|f_ring|f_pinky)(?P<n>\\d)(?P<side>left|right)$'), ('english', '^(?:(?P<side>l|r|left|right)[._ -])?(?:hand[._ -]?|finger[._ -]?)?(?P<digit>thumb|index|middle|ring|pinky|little|mid|f_index|f_middle|f_ring|f_pinky)[._ -]?(?P<n>\\d{1,2})(?:[._ -](?P<side2>l|r|left|right))?$')]
# Rust regex Unicode 16 Decimal_Number ranges, independent of host Python Unicode version.
FINGER_DECIMAL_RANGES = [(48, 57), (1632, 1641), (1776, 1785), (1984, 1993), (2406, 2415), (2534, 2543), (2662, 2671), (2790, 2799), (2918, 2927), (3046, 3055), (3174, 3183), (3302, 3311), (3430, 3439), (3558, 3567), (3664, 3673), (3792, 3801), (3872, 3881), (4160, 4169), (4240, 4249), (6112, 6121), (6160, 6169), (6470, 6479), (6608, 6617), (6784, 6793), (6800, 6809), (6992, 7001), (7088, 7097), (7232, 7241), (7248, 7257), (42528, 42537), (43216, 43225), (43264, 43273), (43472, 43481), (43504, 43513), (43600, 43609), (44016, 44025), (65296, 65305), (66720, 66729), (68912, 68921), (68928, 68937), (69734, 69743), (69872, 69881), (69942, 69951), (70096, 70105), (70384, 70393), (70736, 70745), (70864, 70873), (71248, 71257), (71360, 71369), (71376, 71395), (71472, 71481), (71904, 71913), (72016, 72025), (72688, 72697), (72784, 72793), (73040, 73049), (73120, 73129), (73552, 73561), (90416, 90425), (92768, 92777), (92864, 92873), (93008, 93017), (93552, 93561), (118000, 118009), (120782, 120831), (123200, 123209), (123632, 123641), (124144, 124153), (124401, 124410), (125264, 125273), (130032, 130041)]
_FINGER_DECIMAL_CLASS = "["+"".join(chr(a)+"-"+chr(b) for a,b in FINGER_DECIMAL_RANGES)+"]"
FINGER_PATTERNS = [(key,re.compile(pattern.replace(r"\d",_FINGER_DECIMAL_CLASS))) for key,pattern in FINGER_PATTERNS]

def detect_fingers_by_name(names, parents, deform=None):
    nn,p,d=validate(names,parents,deform)
    return _detect_named(nn,p,d,ancestors)

def _detect_named(nn,p,d,walk):
    for family,pattern in FINGER_PATTERNS:
        groups={}
        for i,name in enumerate(nn):
            if not d[i]:continue
            m=pattern.fullmatch(name)
            if m is None:continue
            x=m.groupdict();side=x.get('side') or x.get('side2')
            if not side:side=''
            side={'left':'l','right':'r','左':'l','右':'r'}.get(side,side)
            digit=x.get('digit') or x.get('palm') or {'01':'index','02':'middle','03':'ring','04':'pinky'}.get(x.get('palmno'))
            digit={'fore':'index','third':'ring','little':'pinky','mid':'middle','人指':'index','親指':'thumb','中指':'middle','薬指':'ring','小指':'pinky'}.get(digit,digit)
            if digit and digit.startswith('f_'):digit=digit[2:]
            if x.get('meta') or x.get('palm') or x.get('palmno'):number=0
            elif x.get('tip'):number=99
            elif family=='biped':
                number_text=x.get('n') or '0'
                number=int(number_text)+1 if number_text.isascii() else 1
            else:
                # Native explicitly folds only fullwidth 0..3 before ASCII
                # integer parsing. Other Unicode decimal captures default to 1.
                number_text=(x.get('n') or '1').translate(str.maketrans('０１２３','0123'))
                number=int(number_text) if number_text.isascii() else 1
            group=groups.setdefault((side,digit),{})
            # Native ordered-map insertion replaces an existing numeric slot.
            # Only eligible entries reach this point; the last input wins.
            group[number]=i
        chains=[];owner={}
        for key,group in groups.items():
            chain=[group[k] for k in sorted(group)]
            if len(chain)>1 and 0 in group:
                if chain[0] not in walk(chain[1],p):chain=chain[1:]
            if chain:
                chains.append(chain)
                owner.update((i,key) for i in chain)
        # Reject each contaminated chain, retaining unrelated valid digits.
        bad=set()
        for i,key in owner.items():
            if key not in bad and any(owner.get(j,key)!=key for j in walk(i,p)):bad.add(key)
        chains=[chain for chain in chains if owner[chain[0]] not in bad]
        chains.sort(key=lambda x:x[0])
        valid=set(groups)-bad
        if any(sum(k[0]==side for k in valid)>=2 for side in ('l','r','')):return chains,family
    return [],''

TORSO_FAMILIES = [{'key': 'mixamo', 'label': 'Mixamo', 'required': ['hips', 'leftupleg'], 'hips': ['hips'], 'spine': ['spine(\\d*)'], 'exclude': [], 'extra': [], 'numeric': True}, {'key': 'arp', 'label': 'Auto-Rig Pro', 'required': ['root\\.x', 'c_root_master\\.x'], 'hips': ['root\\.x'], 'spine': ['(?:c_)?spine_(\\d+)(?:_bend)?\\.x'], 'exclude': [], 'extra': ['c_root_master\\.x', 'c_spine_01\\.x', 'c_spine_02\\.x', 'c_spine_03\\.x'], 'numeric': True}, {'key': 'arp_export', 'label': 'Auto-Rig Pro (export)', 'required': ['root\\.x', 'spine_01\\.x'], 'hips': ['root\\.x'], 'spine': ['spine_(\\d+)\\.x'], 'exclude': ['c_root_master\\.x'], 'extra': [], 'numeric': True}, {'key': 'daz_g9', 'label': 'DAZ Genesis 9', 'required': ['spine4', 'l_thigh'], 'hips': ['pelvis'], 'spine': ['spine(\\d+)'], 'exclude': [], 'extra': ['hip', 'spine1', 'spine2', 'spine3', 'spine4'], 'numeric': True}, {'key': 'daz_g8', 'label': 'DAZ Genesis 3/8', 'required': ['abdomenlower', 'chestupper'], 'hips': ['pelvis'], 'spine': ['abdomenlower', 'abdomenupper', 'chestlower', 'chestupper'], 'exclude': [], 'extra': ['hip', 'abdomenlower', 'abdomenupper', 'chestlower', 'chestupper'], 'numeric': False}, {'key': 'rigify', 'label': 'Rigify (DEF)', 'required': ['def-spine', 'def-spine\\.001'], 'hips': ['def-spine'], 'spine': ['def-spine\\.001', 'def-spine\\.002', 'def-spine\\.003'], 'exclude': [], 'extra': [], 'numeric': False}, {'key': 'def_words', 'label': 'DEF_ words (Rigify-derived)', 'required': ['def_hips', 'def_spine 1'], 'hips': ['def_hips'], 'spine': ['def_spine (\\d+)'], 'exclude': [], 'extra': [], 'numeric': True}, {'key': 'mmd', 'label': 'MMD', 'required': ['下半身', '上半身'], 'hips': ['下半身'], 'spine': ['上半身(\\d*)'], 'exclude': [], 'extra': ['腰', '上半身', '上半身2'], 'numeric': True}, {'key': 'biped', 'label': '3ds Max Biped', 'required': ['bip\\d+[ _-]pelvis'], 'hips': ['bip\\d+[ _-]pelvis'], 'spine': ['bip\\d+[ _-]spine(\\d*)'], 'exclude': [], 'extra': [], 'numeric': True}, {'key': 'unreal', 'label': 'Unreal / MetaHuman', 'required': ['spine_01', 'thigh_l'], 'hips': ['pelvis'], 'spine': ['spine_(\\d+)'], 'exclude': [], 'extra': [], 'numeric': True}, {'key': 'cc', 'label': 'Character Creator', 'required': ['cc_base_hip', 'cc_base_waist'], 'hips': ['cc_base_pelvis'], 'spine': ['cc_base_waist', 'cc_base_spine01', 'cc_base_spine02'], 'exclude': [], 'extra': ['cc_base_hip', 'cc_base_waist', 'cc_base_spine01', 'cc_base_spine02'], 'numeric': False}, {'key': 'vrm', 'label': 'VRoid / VRM', 'required': ['j_bip_c_hips'], 'hips': ['j_bip_c_hips'], 'spine': ['j_bip_c_spine', 'j_bip_c_chest', 'j_bip_c_upperchest'], 'exclude': [], 'extra': [], 'numeric': False}, {'key': 'secondlife', 'label': 'Second Life', 'required': ['mpelvis', 'mtorso'], 'hips': ['mpelvis'], 'spine': ['mspine1', 'mspine2', 'mtorso', 'mspine3', 'mspine4', 'mchest'], 'exclude': [], 'extra': [], 'numeric': False}, {'key': 'roblox', 'label': 'Roblox R15', 'required': ['lowertorso', 'uppertorso'], 'hips': ['lowertorso'], 'spine': ['uppertorso'], 'exclude': [], 'extra': [], 'numeric': False}, {'key': 'humanik', 'label': 'HumanIK / Ready Player Me', 'required': ['hips', 'spine1', 'leftupleg'], 'hips': ['hips'], 'spine': ['spine(\\d*)'], 'exclude': [], 'extra': [], 'numeric': True}, {'key': 'rigify_metarig', 'label': 'Rigify metarig', 'required': ['spine\\.001', 'thigh\\.l'], 'hips': ['spine'], 'spine': ['spine\\.(\\d+)'], 'exclude': ['def-spine'], 'extra': [], 'numeric': True}, {'key': 'english', 'label': 'English generic', 'required': ['(l|r|left|right)?[._ -]?(thigh|upleg|upperleg|upper_leg|up_leg)[._ -]?(l|r|left|right)?(\\.\\d+)?'], 'hips': ['hips', 'pelvis', 'hip'], 'spine': ['waist|spine\\d*|spine[._-]\\d+|chest\\d*|chest[._-]\\d|upper_?chest|torso|torso[._-](lower|upper)|lower_?torso|upper_?torso'], 'exclude': [], 'extra': [], 'numeric': False}]

def ancestors(i,p):
    out=[];i=p[i]
    while i>=0:out.append(i);i=p[i]
    return out

def detect_spine(names, parents, deform, segments=None):
    nn,p,d=validate(names,parents,deform)
    if __package__:
        from .spine_mapping import detect_spine_owned
    else:
        from .spine_mapping import detect_spine_owned
    return detect_spine_owned(names,nn,p,d,segments)


def _detect_spine(names,nn,p,d):
    # Shared mapping body; ordinary callers retain validate() above. The public
    # bind adapter supplies checked lazy parent indexing for native error order.
    result=dict(family='',label='',pelvis=-1,spine=[],neck=[],head=-1,verified=False,reason='no known family signature')
    matches=lambda pattern,deforming=True:[i for i,s in enumerate(nn) if (d[i] or not deforming) and re.fullmatch(pattern,s)]
    family=next((f for f in TORSO_FAMILIES if all(matches(x,False) for x in f['required']) and not any(matches(x,False) for x in f['exclude'])),None)
    if family is None:return result
    result.update(family=family['key'],label=family['label'])
    hips=[i for i,s in enumerate(nn) if d[i] and any(re.fullmatch(x,s) for x in family['hips'])]
    if len(hips)>1:raise ValueError('multiple pelvis candidates remain unqualified')
    if not hips and family['key']=='english':raise ValueError('pelvis inference from leg branches remains unqualified')
    if not hips:
        result['reason']='signature matched but the pelvis bone is missing or does not deform';return result
    pelvis=hips[0];result['pelvis']=pelvis
    spine=[]
    if family['numeric']:
        found=[]
        for i,s in enumerate(nn):
            m=re.fullmatch(family['spine'][0],s)
            if d[i] and m:found.append((int(m.group(1) or 0),i))
        if len({x[0] for x in found})!=len(found):raise ValueError('duplicate numbered torso candidates remain unqualified')
        spine=[i for _,i in sorted(found)]
    else:
        for pattern in family['spine']:
            ids=matches(pattern)
            if family['key']=='english':
                ids.sort(key=lambda i:len(ancestors(i,p)))
            elif len(ids)>1:raise ValueError('ambiguous torso pattern')
            spine.extend(ids if family['key']=='english' else ids[:1])
    if family['key']=='english':
        spine=[i for i in spine if i!=pelvis]
        # Generic route walks the named chain from leaf toward its ancestors.
        spine=[i for j,i in enumerate(spine) if not j or spine[j-1] in ancestors(i,p)]
        if not spine:
            result['reason']='torso bones do not form a chain';return result
    # The torso ends at the common shoulder/arm ancestry, before neck/head.
    arms=[i for i,s in enumerate(nn) if re.search(r'shoulder|clavicle|collar',s) or (re.search(r'(^|[._ -]|left|right)(upper[._ -]?)?arm(\d|[._ -]|$)',s) and 'forearm' not in s and 'armature' not in s)]
    if arms:
        common=set(ancestors(arms[0],p))
        for i in arms[1:]:common.intersection_update(ancestors(i,p))
        cut=[j for j,i in enumerate(spine) if i in common]
        if cut:spine=spine[:max(cut)+1]
    result['spine']=spine
    if not spine:
        count=sum(x==pelvis for x in p)
        result.update(verified=count>=2,reason='no torso bones; pelvis only' if count>=2 else 'no torso bones and the pelvis parents nothing');return result
    if family['key']=='english':
        first=spine[0]
        if p[first]!=pelvis and p[pelvis]!=first and not (p[pelvis]>=0 and p[first]==p[pelvis]):
            result['reason']=f'hierarchy: "{names[pelvis]}" is not adjacent to "{names[first]}"';return result
        check=spine
    elif family['extra']:
        check=[]
        for pattern in family['extra']:
            ids=matches(pattern,False)
            if not ids:break
            check.append(ids[0])
        if len(check)<2:result['reason']='verify chain bones missing';return result
    else:check=[pelvis]+spine
    for a,b in zip(check,check[1:]):
        if a not in ancestors(b,p):
            result['reason']=f'hierarchy: "{names[a]}" is not an ancestor of "{names[b]}"';return result
    necks=[i for i,s in enumerate(nn) if d[i] and re.search(r'neck|collo|kubi|首',s)]
    neck=[]
    for i in necks:
        if not neck or neck[-1] in ancestors(i,p) or p[i]<0:neck.append(i)
    heads=[i for i,s in enumerate(nn) if d[i] and re.search(r'head|skull|頭',s) and s not in ('headtop','headend')]
    if family['key']=='vrm':
        exact_head=matches('j_bip_c_head')
        if len(exact_head)==1:heads=exact_head
        elif heads:heads=heads[:1] # Named VRM fallback retains skeleton order.
    if len(heads)>1:raise ValueError('multiple named heads remain unqualified')
    head=heads[0] if heads else -1
    if head<0 or not neck:
        # Geometric-free fallback only follows one eligible child at each step.
        def child(i):
            ids=[j for j,v in enumerate(p) if v==i and d[j]]
            if len(ids)>1:raise ValueError('branching head fallback remains unqualified')
            return ids[0] if ids else -1
        c1=child(spine[-1]);c2=child(c1) if c1>=0 else -1
        if c1>=0 and c2>=0:
            if not neck:neck=[c1]
            if head<0:head=c2
    result.update(neck=neck,head=head,verified=True,reason='');return result


def _public_named_fingers(nn,parents,deform):
    """Typed native entry owns extraction/normalization; index only visited parents."""
    from . import _bvt_solver as runtime
    class Parents(list):
        def __getitem__(self,index):
            if index<0 or index>=len(self):
                raise runtime.PanicException(f'index out of bounds: the len is {len(self)} but the index is {index}')
            return super().__getitem__(index)
    p=Parents(parents)
    if deform is None:d=[True]*len(nn)
    else:
        if len(deform)<len(nn):raise runtime.PanicException(f'index out of bounds: the len is {len(deform)} but the index is {len(deform)}')
        d=deform
    def walk(i,p):
        if i>=len(p):p[i] # original checks the starting index even for zero parents
        for _ in range(len(p)):
            i=p[i]
            if i<0:break
            yield i
            if i>=len(p):p[i]
    return _detect_named(nn,p,d,walk)
