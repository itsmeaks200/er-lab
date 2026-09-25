"""Text normalisation engine (frozen; identical for train and test).
Ported from the notebook: dependency-free Unicode-name transliteration, name/address representations,
learned synonym maps (NAME_MAP / ADDR_MAP are filled from fit-fold positives and shipped to workers)."""
import re
import unicodedata
from collections import Counter

_DOMAIN_RE = re.compile(r'\.(?:com|net|org|in|co|biz|info|io|us|fr)\b|^[@#][a-z0-9]|www\.')


_SPECIAL = {'œ': 'oe', 'æ': 'ae', 'ß': 'ss', 'ø': 'o', 'ł': 'l', 'đ': 'd', 'ð': 'd', 'þ': 'th', 'ı': 'i', 'ĳ': 'ij',
            '’': "'", '‘': "'", '´': "'", '`': "'", '–': '-', '—': '-', '“': '"', '”': '"', '«': '"', '»': '"',
            '।': ' ', '॥': ' '}
_VOWELS = {'a': 'a', 'aa': 'a', 'i': 'i', 'ii': 'i', 'u': 'u', 'uu': 'u', 'e': 'e', 'ee': 'e',
           'ai': 'ai', 'o': 'o', 'oo': 'o', 'au': 'au'}
_REP = re.compile(r'(.)\1+')


def _indic_letter(rest):
    w = rest.lower().replace('-', ' ').split()
    if not w:
        return ''
    if w[0] == 'vocalic':
        return 'ri' if len(w) > 1 and w[1].startswith('r') else 'li'
    if w[0] in ('candra', 'short', 'independent', 'small', 'capital', 'final', 'medial', 'initial') and len(w) > 1:
        w = w[1:]
    tok = w[-1]
    if tok in _VOWELS:
        return _VOWELS[tok]
    if len(tok) > 1 and tok.endswith('a'):
        tok = tok[:-1]                       # consonant: drop inherent vowel (KA -> k, TTA -> t)
    return _REP.sub(r'\1', tok)


def _char_map(ch):
    if ch in _SPECIAL:
        return _SPECIAL[ch]
    cat = unicodedata.category(ch)
    name = unicodedata.name(ch, '')
    if cat in ('Mn', 'Mc', 'Me', 'Cf'):          # combining marks, Indic vowel signs, ZWJ...
        if 'VOWEL SIGN' in name:
            v = name.split('VOWEL SIGN ', 1)[1].lower().split()
            if v and v[0] == 'vocalic':
                return 'ri'
            return _VOWELS.get(v[-1], '') if v else ''
        if name.endswith('ANUSVARA') or name.endswith('CANDRABINDU'):
            return 'n'
        if name.endswith('VISARGA'):
            return 'h'
        return ''
    if cat == 'Nd':
        d = unicodedata.decimal(ch, None)
        return str(d) if d is not None else ' '
    base = ''.join(c for c in unicodedata.normalize('NFKD', ch) if not unicodedata.combining(c))
    if base and base.isascii():
        return base.lower()
    if cat.startswith('L') and ' LETTER ' in name:
        return _indic_letter(name.split(' LETTER ', 1)[1])
    if cat[0] in 'PSZ':
        return ' '
    return ch


class _TranslitTable(dict):
    def __missing__(self, key):
        v = _char_map(chr(key))
        self[key] = v
        return v


TRANSLIT = _TranslitTable()


def translit(s):
    return s if s.isascii() else s.translate(TRANSLIT)


LEGAL_FORMS = {
    'inc', 'incorporated', 'llc', 'llp', 'pllc', 'lp', 'ltd', 'limited', 'pvt', 'private', 'privet', 'praivet', 'prayvet',
    'corp', 'corporation', 'co', 'cos', 'company', 'plc', 'pte', 'opc', 'gmbh', 'ag', 'sa', 'sas', 'sasu', 'sarl', 'eurl',
    'sci', 'snc', 'scs', 'scop', 'selarl', 'sprl', 'bv', 'nv', 'srl', 'spa',
    'the', 'and', 'of', 'et', 'de', 'du', 'des', 'la', 'le', 'les', 'l', 'd',
}
ADDR_HAND_MAP = {
    'street': 'st', 'str': 'st', 'strt': 'st', 'avenue': 'ave', 'av': 'ave', 'avn': 'ave', 'road': 'rd', 'drive': 'dr',
    'drv': 'dr', 'lane': 'ln', 'court': 'ct', 'crt': 'ct', 'place': 'pl', 'boulevard': 'blvd', 'boul': 'blvd',
    'bd': 'blvd', 'blv': 'blvd', 'highway': 'hwy', 'parkway': 'pkwy', 'pky': 'pkwy', 'circle': 'cir', 'terrace': 'ter',
    'terr': 'ter', 'trail': 'trl', 'square': 'sq', 'suite': 'ste', 'apartment': 'apt', 'appt': 'apt', 'building': 'bldg',
    'floor': 'fl', 'flr': 'fl', 'north': 'n', 'south': 's', 'east': 'e', 'west': 'w', 'northeast': 'ne',
    'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw', 'mount': 'mt', 'fort': 'ft', 'route': 'rte',
    'expressway': 'expy', 'freeway': 'fwy', 'junction': 'jct', 'heights': 'hts', 'center': 'ctr', 'centre': 'ctr',
    'point': 'pt', 'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5', 'sixth': '6',
    'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10', 'number': 'no', 'nr': 'near', 'opposite': 'opp',
    'sector': 'sec', 'sect': 'sec', 'district': 'dist', 'r': 'rue', 'chemin': 'ch', 'allee': 'all', 'impasse': 'imp',
    'faubourg': 'fbg', 'saint': 'st', 'sainte': 'ste', 'residence': 'res', 'batiment': 'bat',
}
ADDR_PLACEHOLDER_TOKENS = {'null', 'none', 'nan', 'na', 'nil', 'unknown', 'undefined', 'notavailable'}
NAME_MAP, ADDR_MAP = {}, {}          # learned from TRAIN fit-fold positive pairs in §6/§7 (empty until then)

_WWW = re.compile(r'\bwww\.')
_TLD = re.compile(r'\.(?:com|net|org|in|co|biz|info|io|us|fr|uk|online|site|store|shop)\b')
_APOS = re.compile(r"['’`´]")
_NON_ALNUM = re.compile(r'[^0-9a-z]+')
_SINGLE_RUN = re.compile(r'\b(?:[a-z] )+[a-z]\b')
_ORDINAL = re.compile(r'\b(\d+)(?:st|nd|rd|th)\b')
_NUM = re.compile(r'\d+')
_VOW = re.compile(r'[aeiouy]')
_TRADE = re.compile(r'\b(?:ta|dba|aka|fka|trading as|doing business as)\b')
_LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't'})


def _merge_singles(m):
    return m.group(0).replace(' ', '')


def name_basic(raw):
    s = translit(raw.lower())
    s = _TLD.sub(' ', _WWW.sub(' ', s))
    s = _APOS.sub('', s.replace('&', ' and '))
    s = _NON_ALNUM.sub(' ', s).strip()
    return _SINGLE_RUN.sub(_merge_singles, s)


def _fix_leet(t):
    if t.isalpha() or t.isdigit():
        return t
    if sum(c.isdigit() for c in t) == 1 and len(t) >= 2:
        return t.translate(_LEET)
    return t


def name_core(basic):
    toks = [NAME_MAP.get(t, t) for t in (_fix_leet(t) for t in basic.split())]
    core = [t for t in toks if t not in LEGAL_FORMS]
    return ' '.join(core if core else toks)


def skel_tok(t):
    if not t or t.isdigit():
        return t
    t = t.replace('ph', 'f').replace('ck', 'k').replace('q', 'k').replace('w', 'v').replace('z', 's').replace('sh', 's')
    return _REP.sub(r'\1', t[0] + _VOW.sub('', t[1:]))


def name_skel(core):
    return ' '.join(skel_tok(t) for t in core.split())


def name_alt(basic):
    """Trade-name part after 't/a', 'dba', 'aka' ... (empty if none)."""
    parts = _TRADE.split(basic)
    return name_core(parts[-1].strip()) if len(parts) > 1 and parts[-1].strip() else ''


def acronym(core):
    t = [x for x in core.split() if not x.isdigit()]
    return ''.join(x[0] for x in t) if len(t) >= 2 else ''


def addr_basic(raw):
    s = translit(raw.lower())
    s = _APOS.sub('', s.replace('&', ' and '))
    s = _ORDINAL.sub(r'\1', _NON_ALNUM.sub(' ', s))
    s = _SINGLE_RUN.sub(_merge_singles, s.strip())
    return ' '.join(t for t in s.split() if t not in ADDR_PLACEHOLDER_TOKENS)


def addr_canon(basic):
    return ' '.join(ADDR_MAP.get(t2, t2) for t2 in (ADDR_HAND_MAP.get(t, t) for t in basic.split()))


def addr_keys(canon):
    """house-number + street-token keys, robust to component re-ordering ('OR, Eugene, 3900 River Rd')."""
    t = canon.split()
    keys = []
    for i, x in enumerate(t[:-1]):
        if x.isdigit() and not t[i + 1].isdigit() and len(t[i + 1]) >= 3:
            keys.append((x.lstrip('0') or '0') + '|' + skel_tok(t[i + 1])[:4])
            if len(keys) >= 3:
                break
    return ' '.join(keys)


def name_flags(raw):
    low = raw.lower()
    b = 0
    if _DOMAIN_RE.search(low):
        b |= 1
    if not raw.isascii():
        b |= 2
    if raw.isupper():
        b |= 4
    if raw[:1] and not raw[:1].isalnum():
        b |= 8
    return b


def first_number(canon):
    m = _NUM.search(canon)
    return int(m.group(0)[:9]) if m else -1


def learn_token_map(left, right, min_count=40, min_prob=0.5, max_diff=2):
    """Data-driven synonym map from positive pairs: token r (noisy side) → token l (S1 side) when r systematically
    replaces l. Keys and targets are kept disjoint (no chains)."""
    pair_c, r_c = Counter(), Counter()
    for a, b in zip(left, right):
        A, B = set(a.split()), set(b.split())
        L, R = A - B, B - A
        if not L or not R or len(L) > max_diff or len(R) > max_diff:
            continue
        w = 1.0 / (len(L) * len(R))
        for r in R:
            r_c[r] += 1
            for l in L:
                pair_c[(r, l)] += w
    mp, targets = {}, set()
    for (r, l), c in sorted(pair_c.items(), key=lambda x: -x[1]):
        if c < min_count:
            break
        if r in mp or l in mp or r in targets or r == l or r.isdigit() or l.isdigit():
            continue
        if c / r_c[r] >= min_prob:
            mp[r] = l
            targets.add(l)
    return mp


_LANDMARK = {'near', 'opp', 'behind', 'beside', 'besides', 'adjacent', 'nr', 'bh', 'infront', 'next'}


def postal_codes(canon):
    """postal-like numbers: 6-digit tokens (PIN) or 5-digit tokens not followed by a street word (ZIP / code postal)."""
    t = canon.split()
    out = []
    for i, x in enumerate(t):
        if x.isdigit() and (len(x) == 6 or (len(x) == 5 and not (i + 1 < len(t) and t[i + 1].isalpha() and len(t[i + 1]) > 2))):
            out.append(x)
    return out


def _nums(c):
    return {x.lstrip('0') or '0' for x in _NUM.findall(c)}


def postal_str(canon):
    return ' '.join(postal_codes(canon))


def addr_set_key(canon):
    return ' '.join(sorted(set(canon.split())))
