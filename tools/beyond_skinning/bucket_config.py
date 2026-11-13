"""Public bind bucket configuration, independent of semantic torso overrides."""
DEFAULT = '2,14,6,14,6'
# Rust str::trim uses Unicode White_Space, unlike Python's additional C0 separators.
SPACE = ' \t\n\r\v\f\x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000'
def parse(text):
    # Bind trims outer whitespace before handing its key to bones_for.
    # Interior spelling remains significant to cache ownership.
    text = DEFAULT if text is None else text.strip(SPACE)
    parts = text.split(',')
    if len(parts) != 5: return text, None
    values = []
    for part in parts:
        part = part.strip(SPACE)
        negative = part.startswith('-')
        if part.startswith(('-', '+')): part = part[1:]
        if not part: return text, None
        value = 0
        limit = (1 << 63) if negative else (1 << 63) - 1
        for digit in part:
            if not '0' <= digit <= '9': return text, None
            value = value * 10 + ord(digit) - 48
            if value > limit: return text, None
        values.append((-value if negative else value) & 255)
    return text, tuple(values)
