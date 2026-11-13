"""Semantic material labels without renaming or changing artist data."""
import unicodedata

ALIASES={
    'hair': ('髪','髮','头发','頭髮','毛发','毛髪','ヘア','머리카락','머리털'),
    'skin': ('肌','皮肤','皮膚','スキン','피부'),
    'face': ('顔','颜','顏','フェイス','얼굴'),
    'eye': ('二重','目','眼','瞳','眉','睫','アイ','눈','눈썹'),
    'mouth': ('口','舌','リップ','입'),
    'teeth': ('歯','齿','齒','치아'),
    'skirt': ('スカート','裙','치마'),
    'shoe': ('靴','鞋','シューズ','신발'),
    'sock': ('袜','襪','ソックス','양말'),
    'glove': ('手袋','手套','장갑'),
}


def semantic_label(name):
    text=unicodedata.normalize('NFKC',str(name)).casefold()
    return ' '.join([text]+[role for role,terms in ALIASES.items() if any(t in text for t in terms)])
