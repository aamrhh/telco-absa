"""
Rule-based Manglish normalizer (adapted from Ariffin & Tiun, 2020).

Six rule categories, applied in order, token by token:
  1. Noise removal             - strip stray punctuation/symbols; mask profanity
  2. Abbreviation expansion    - handled via dictionary lookup (manglish_dictionary.csv)
  3. Compound word handling    - reduplicated words (e.g. "mahal2" -> "mahal-mahal")
  4. Interjection standardisation - handled via dictionary lookup (manglish_dictionary.csv)
  5. Typographical correction  - fixes ANY misspelling, not just elongation, e.g.
                                  "broblom" -> "problem", "intenet" -> "internet",
                                  "sloww" -> "slow" - via edit-distance spellchecking
  6. Slang/colloquial normalisation - handled via dictionary lookup (manglish_dictionary.csv)

DIVISION OF WORK (v3, 19 Oct):
  NORMALIZER (this file) = SPELLING.  Short forms, abbreviations, typos, stretched letters, brand
      misspellings  ->  ONE standard word   (skrg -> sekarang, thx -> thank you, mcm -> macam).
      Where the rules live:
        - SPELLING_FAMILIES / SPELLING_VARIANTS   (in this file: patterns and 8+ spellings per word)
        - data/processed/normalizer_short_forms.csv   (one line per short form; edit freely)
        - data/processed/malay_slang.csv              (your Malay slang list; decision = use / skip)
        - LEAVE_ALONE (slurs, bengong, lol, bro)  and  FILLERS (la, lah are dropped)
      The spellchecker (English edit-distance guessing) is OFF by default: USE_SPELLCHECKER = False.
  DICTIONARY (manglish_dictionary.csv) = MEANING.  The standard word -> English / Malay
      (sekarang -> now / sekarang).  No short forms or typos in it.

Pipeline order: normalize_text() FIRST, then dictionary_mapper.map_text().
"""

import re
from functools import lru_cache
from pathlib import Path

try:
    from spellchecker import SpellChecker
    _SPELLCHECKER = SpellChecker()
except ImportError:  # pragma: no cover
    _SPELLCHECKER = None  # pip install pyspellchecker

try:
    from dictionary_mapper import load_dictionary
    _dictionary_terms = load_dictionary().keys()
    _KNOWN_TERMS = set(_dictionary_terms)
    # Multi-word dictionary entries (e.g. "macam siput") need their
    # INDIVIDUAL words protected too, not just the full phrase - otherwise
    # the normalizer can corrupt one word of the phrase (e.g. "siput" ->
    # "put") before the dictionary mapper ever gets a chance to recognise
    # the phrase as a whole in the next pipeline step.
    for _term in _dictionary_terms:
        _KNOWN_TERMS.update(_term.split())
except Exception:  # pragma: no cover - dictionary CSV not found / not run standalone
    _KNOWN_TERMS = set()
# Note: telco brand/product names (celcom, digi, maxis, umobile, etc.) and
# their common misspellings now live directly in manglish_dictionary.csv
# under category "brand_name" - so they're automatically included in
# _KNOWN_TERMS above, which means normalize_token() leaves them alone here,
# and dictionary_mapper.py corrects any misspelled variant to the proper
# brand spelling in the next pipeline step. One source of truth, not two.

# ---------------------------------------------------------------------------
# Rule category 1: Noise removal
# ---------------------------------------------------------------------------
# Profanity is masked here rather than translated in the dictionary, since
# these terms don't have a meaningful "standard English/Malay" form.
PROFANITY_TERMS = set()     # nothing is masked any more: slurs are kept as written (see LEAVE_ALONE)
PROFANITY_MASK = "[MASKED]"

# Words that must NEVER be changed or translated: slurs stay as standard Malay (English and Malay
# version both show the same word), bengong stays bengong, lol and bro stay as they are.
LEAVE_ALONE = {"cibai", "kimak", "celaka", "sial", "lancau", "buto", "kontol", "pantat", "mampus",
               "hanat", "barua", "bengong", "lol", "bro"}
# Filler words: dropped (the normalizer returns an empty string; step 3 removes empty tokens)
FILLERS = {"la", "lah"}

# Leftover noise: stray symbols not already stripped by the cleaning step
# (kept here as a safety net for the normalizer stage)
NOISE_PATTERN = re.compile(r"[^\w\s\-]")


def remove_noise(token: str) -> str:
    if token in PROFANITY_TERMS:
        return PROFANITY_MASK
    return NOISE_PATTERN.sub("", token)


# ---------------------------------------------------------------------------
# Rule category 3: Compound word handling (reduplication fallback)
# Catches the general Malay "word2" -> "word word" shorthand for any
# reduplicated word not already an explicit entry in the dictionary CSV.
# ---------------------------------------------------------------------------
REDUPLICATION_PATTERN = re.compile(r"^([a-z]+)2$")


def handle_compound_word(token: str) -> str:
    match = REDUPLICATION_PATTERN.match(token)
    if match:
        root = match.group(1)
        if root == "rm":
            return token  # "rm2" means RM2 (two ringgit), not "rm-rm"
        return f"{root} {root}"
    return token


# ---------------------------------------------------------------------------
# Rule category 5: Typographical correction
# ---------------------------------------------------------------------------
# A Malay word guard so the English spellchecker doesn't "correct" valid
# Malay tokens into English words (e.g. "langsung" should NOT become "lansing").
# Uses malaya's full Malay word list when available (it's already installed
# elsewhere in the pipeline, for tokenization); falls back to a small
# safety-net list here so this file still runs standalone without malaya/torch.
try:
    from malaya.dictionary import MALAY_WORDS as _MALAY_WORDS
except ImportError:  # pragma: no cover - fallback for standalone/demo use
    _MALAY_WORDS = set()

_COMMON_MALAY_WORDS = _MALAY_WORDS | {
    "dan", "atau", "yang", "ini", "itu", "saya", "awak", "kami", "kita",
    "boleh", "tidak", "ada", "akan", "sudah", "belum", "pergi", "datang",
    "makan", "minum", "rumah", "kereta", "telefon", "bagus", "buruk",
    "cepat", "lambat", "besar", "kecil", "banyak", "sikit", "hari", "bulan",
    "tahun", "pukul", "pagi", "petang", "malam", "terima", "kasih",
    "tolong", "maaf", "sila", "masalah", "aplikasi", "perkhidmatan",
    "pelanggan", "syarikat", "juga", "dengan", "untuk", "kerana", "macam",
    "orang", "guna", "buat", "baik", "lagi", "mesti", "harap", "semoga",
}

# Elongation collapse runs first (cheap + deterministic) so the spellchecker
# sees "baddddd" as "badd" -> closer match -> still routes to "bad".
ELONGATION_PATTERN = re.compile(r"(.)\1{2,}")


def correct_elongation(token: str) -> str:
    # Only for words made of letters. Numbers like "1000" or "2000gb" must
    # never be touched (a run of zeros is not a stretched word).
    if not any(c.isalpha() for c in token) or any(c.isdigit() for c in token):
        return token
    # "tooooo" -> "too" (keep two letters if that is a real word),
    # otherwise "slowww" -> "slow", "jeee" -> "je" (collapse to one letter).
    one = ELONGATION_PATTERN.sub(r"\1", token)
    if (one in _KNOWN_TERMS or one in LEAVE_ALONE or one in FILLERS or one in SPELLING_SHORT_FORMS) and one != token:
        return one                                  # "lolll" -> "lol", "tqqq" -> "tq", "laaa" -> "la"
    two = ELONGATION_PATTERN.sub(r"\1\1", token)
    if (_SPELLCHECKER is not None and two != token
            and token.isalpha() and two not in _SPELLCHECKER.unknown([two])):
        return two
    return ELONGATION_PATTERN.sub(r"\1", token)


# A genuine typo is RARE (one person mistyped it once). A word that appears
# many times across the 82k reviews is almost certainly real slang, a Malay
# short form or a telco/tech term (telco, takde, utk, sms, vpn...), so the
# English spellchecker must leave it alone. The step-3 script fills this in
# with real word counts from the data before normalizing.
MIN_FREQ_TO_KEEP = 8
# Switch for the spellchecker's GUESSING step (rule 5).  False = "normalizer without spellchecker":
# the other rules still run (stretched letters, spelling families, Malay slang list).
USE_SPELLCHECKER = False        # OFF by default (it turned plis into plus, kne into knee ...). True = the "with spellchecker" version
MIN_FREQ_FOR_CORRECTION = 20    # a suggested correction must appear at least this often in the reviews
_TOKEN_FREQ = {}


def set_token_frequencies(counts) -> None:
    global _TOKEN_FREQ
    _TOKEN_FREQ = dict(counts)
    normalize_token.cache_clear()


def malay_list_size() -> int:
    return len(_COMMON_MALAY_WORDS)


def _is_valid_word(word: str) -> bool:
    """A real word: in the dictionary, a Malay word, frequent in the data, or an English word."""
    if word in _KNOWN_TERMS or word in _COMMON_MALAY_WORDS:
        return True
    if _TOKEN_FREQ.get(word, 0) >= MIN_FREQ_TO_KEEP:
        return True
    return _SPELLCHECKER is not None and word not in _SPELLCHECKER.unknown([word])


_RUN = re.compile(r"(.)\1+")


def collapse_double_letters(token: str) -> str:
    """
    A stretched word with only TWO repeated letters ("bodohh", "noo", "stupidd", "lamaa"):
    the 3+ rule above does not touch it. Collapse the repeat ONLY when the long form is not a
    real word and the short form is  ("goood" stays for the dictionary, "all", "see", "too" stay).
    """
    if len(token) < 3 or not token.isalpha() or not _RUN.search(token) or _is_valid_word(token):
        return token
    for m in reversed(list(_RUN.finditer(token))):               # stretching usually happens at the end
        candidate = token[:m.start()] + m.group(1) + token[m.end():]
        if _is_valid_word(candidate):
            return candidate
    candidate = _RUN.sub(r"\1", token)
    return candidate if _is_valid_word(candidate) else token


def correct_typo(token: str) -> str:
    """
    Edit-distance based spelling correction (rule category 5), for
    misspellings the dictionary CSV has never seen - not just elongation,
    e.g. "broblom" -> "problem", "netwerk" -> "network", "custome" -> "customer".
    """
    if _SPELLCHECKER is None or not USE_SPELLCHECKER:
        return token
    if _TOKEN_FREQ.get(token, 0) >= MIN_FREQ_TO_KEEP:
        return token  # frequent word = real word/slang, not a typo
    if not token.isalpha() or len(token) < 3:
        return token
    if token in _COMMON_MALAY_WORDS:
        return token
    if token not in _SPELLCHECKER.unknown([token]):
        return token  # already a recognised English word
    # Data-backed correction: only accept a suggested word that people REALLY use in these
    # reviews (frequency counts come from the step-3 script).  "kne" -> "knee" is rejected
    # because "knee" is almost never written in telco reviews; "intenet" -> "internet" is accepted.
    if _TOKEN_FREQ:
        suggestions = _SPELLCHECKER.candidates(token) or set()
        usable = {c: _TOKEN_FREQ.get(c, 0) for c in suggestions if _TOKEN_FREQ.get(c, 0) >= MIN_FREQ_FOR_CORRECTION}
        if not usable:
            return token                       # no suggestion people actually use: leave the word alone
        return max(usable, key=usable.get)
    correction = _SPELLCHECKER.correction(token)   # no frequency data (standalone use): old behaviour
    return correction if correction else token


# ---------------------------------------------------------------------------
# Rule category 5b: spelling families (the normalizer fixes SPELLING, the
# dictionary only TRANSLATES the standard word afterwards)
# ---------------------------------------------------------------------------
# Stretched or shortened spellings of a few very common slang words are mapped to ONE
# standard word (sangat / teruk / pastu / so).  Step 4 (dictionary) then turns that
# standard word into English / Malay.  Runs of exactly two letters (saangat, sangatt)
# are covered here, because the general elongation rule only collapses runs of 3+.
SPELLING_FAMILIES = [
    (re.compile(r"^s+a+n+g+a+t+$"), "sangat"),     # sangat, saangat, sangattt, saaaaangat ...
    (re.compile(r"^s+o{2,}$"), "so"),               # soo, sooo ... soooooooooo
    (re.compile(r"^ter[uo]k+s*$"), "teruk"),        # terok, teruk, teroks, terukss, terukk ...
    (re.compile(r"^pastu+$"), "selepas itu"),       # pastu, pastuu ...  (= selepas itu, "after that")
    (re.compile(r"^ma*c+[ae]*m+$"), "macam"),       # mcm, mcmm, macm, mcam, macem, mcem, macamm, macammm ...
    (re.compile(r"^t+a+k+$|^t+k+$"), "tidak"),      # tk, tkk, tak, takk, takkk ...  (standard word: tidak)
    (re.compile(r"^p+l+[ie]*a*[sz]+e*$"), "please"),   # pls, plz, plis, pliss, plsss, pleas, pleaze ...
    (re.compile(r"^(?:t+h+n?x+|t+h+a+n+x+|t+q+|d+q+)$|^thank(?:y+o+u+|u+)$"), "thank you"),  # thx, thxx, thanx, tq, dq, thanku, thankyou
    (re.compile(r"^n+a+k+$|^n+k+$"), "nak"),        # nk, nkk, nak, nakk, nakkk ...
    (re.compile(r"^j+e+(?:r+|k+|h+)?$"), "sahaja"), # je, jee, jer, jerr, jek ...    (awak je boleh = only you can)
]
SPELLING_SHORT_FORMS = {
    "sgt": "sangat", "sngt": "sangat", "sangt": "sangat", "sgnt": "sangat", "sgat": "sangat",
    "pstu": "pastu", "lepastu": "pastu", "lpastu": "pastu", "pahtu": "pastu", "patu": "pastu",
}

# More spelling families (standard word -> its spellings). Every word has AT LEAST 8 spellings:
# short forms, dropped vowels, stretched letters and common typos. Edit freely; one line per word.
SPELLING_VARIANTS = {
    "sangat":  "sgt sngt sangt sgnt sgat sgtt sgttt sangatt sangattt",
    "selepas itu": "pastu pstu lepastu lpastu lpstu pahtu patu pastuu pasttu lepasttu lepahtu",
    "sebab":   "sbb sbab sbap sebb sebap sebabb sebabbb sebabbbb sbbb",
    "dengan":  "dgn dgan dengn dngn dengang denagn dengann dengannn dgnn",
    "tapi":    "tp tpi tapii tapiii tapiiii tpii tapy tappi tapee",
    "untuk":   "utk untk utuk untok untukk untukkk untuq uuntuk",
    "kalau":   "kalo klu klo kalu kalauu kalauuu kalaw kalow",
    "sampai":  "smpai sampe smpe smpi sampei sampaii sampaiii sampay samapi",
    "banyak":  "byk bnyk banyk bnyak byak banyakk banyakkk banyaak bnyakk",
    "boleh":   "blh bleh bole blehh bolehh boleeh bolehhh bley boley bolh bolee",
    "sikit":   "skit sket sikitt sikittt sikiit sikt cikit ckit sikitttt",
    "betul":   "btl betol btul betoi betui btol betull betulll betullll",
    "mahal":   "mhl mhal mahl mahall mahalll mahallll mahaal mahhal maahal",
    "bagus":   "bgs bagos bgus baguss bagusss bagussss baguus baagus bagoss",
    "sudah":   "dah dh dahh sdh dahhh dahhhh daah daaah dha",
    "lagi":    "lg lgi lagii lagiii lagiiii lgii lagy laggi lagie",
    "juga":     "pun pon punn punnn puun poon ponn ponnn puon jg jugak juger jugakk jugakkk jugaa jugaaa",
    "sahaja":     "je jee jer jerr jerrr jek jeek jeeek jeeekk ja jaa shj",
}
for _standard, _spellings in SPELLING_VARIANTS.items():
    for _spelling in _spellings.split():
        SPELLING_SHORT_FORMS[_spelling] = _standard

# The short-form list (one line per short form / typo -> standard word).  Edit the CSV, not the code.
SHORT_FORMS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "normalizer_short_forms.csv"


def load_short_forms(path=SHORT_FORMS_PATH) -> dict:
    out = {}
    try:
        import csv
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                v, s = row["variant"].strip().lower(), row["standard"].strip().lower()
                if v and s and v != s:
                    out[v] = s
    except FileNotFoundError:  # file not there: only the built-in families below are used
        pass
    return out


SPELLING_SHORT_FORMS.update(load_short_forms())


# ---------------------------------------------------------------------------
# Rule category 6: Malay slang / short forms -> standard Malay  (data/processed/malay_slang.csv)
# ---------------------------------------------------------------------------
# blh -> boleh, tgk -> tengok, byk -> banyak ...  Only rows with decision = "use" are applied.
# Set USE_MALAY_SLANG = False to switch this step off and compare (it is one of the text versions to test).
USE_MALAY_SLANG = True
MALAY_SLANG_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "malay_slang.csv"


def load_malay_slang(path=MALAY_SLANG_PATH) -> dict:
    out = {}
    try:
        import csv
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("decision", "").strip().lower() == "use":
                    out[row["slang"].strip().lower()] = row["standard_malay"].strip().lower()
    except FileNotFoundError:  # file not there yet: the step simply does nothing
        pass
    return out


_MALAY_SLANG = load_malay_slang()


def standardize_spelling(token: str):
    """Returns the standard word for a known spelling family, or None."""
    if token in SPELLING_SHORT_FORMS:
        return SPELLING_SHORT_FORMS[token]
    for pattern, standard in SPELLING_FAMILIES:
        if pattern.match(token):
            return standard
    return None


@lru_cache(maxsize=None)
def normalize_token(token: str) -> str:
    """
    Apply the algorithmic normalizer rules to a single token, in order.
    Tokens already known to the dictionary (manglish_dictionary.csv) are
    passed through untouched - dictionary_mapper.py resolves those next.

    Cached: on a large dataset the same word (e.g. "topup", "the", "customer")
    appears thousands of times, and the spellchecker correction step is the
    slow part - caching means each distinct word is only corrected once,
    no matter how many reviews it appears in.
    """
    if not token:
        return token

    original = token
    token = token.lower().strip()

    # 1. Fillers (la, lah) are dropped; slurs / bengong / lol / bro are never touched
    fixed = _fixed_form(token)
    if fixed is not None:
        return fixed

    # 2. Leave known dictionary terms alone; dictionary_mapper.py handles them next
    if token in _KNOWN_TERMS:
        return original

    # 3. Malay slang list (data/processed/malay_slang.csv): blh -> boleh, tgk -> tengok ...
    if USE_MALAY_SLANG and token in _MALAY_SLANG:
        return _MALAY_SLANG[token]

    # 4. Noise removal (stray symbols)
    token = remove_noise(token)
    if token == PROFANITY_MASK:
        return token
    fixed = _fixed_form(token)
    if fixed is not None:
        return fixed
    if token in _KNOWN_TERMS:
        return token

    # 5. Reduplication fallback:  mahal2 -> mahal-mahal
    compound = handle_compound_word(token)
    if compound != token:
        return compound

    # 6. Stretched letters, then (only if switched on) the spellchecker
    elongation_fixed = correct_elongation(token)
    if elongation_fixed != token:
        token = elongation_fixed
        fixed = _fixed_form(token)          # tqqq -> tq -> thank you,  jeee -> je -> sahaja,  lolll -> lol
        if fixed is not None:
            return fixed
        if token in _KNOWN_TERMS:
            return token
    doubled = collapse_double_letters(token)       # bodohh -> bodoh, noo -> no, stupidd -> stupid
    if doubled != token:
        fixed = _fixed_form(doubled)               # camm -> cam -> macam
        return fixed if fixed is not None else doubled
    typo_fixed = correct_typo(token)
    if typo_fixed != token:
        return typo_fixed
    if elongation_fixed != original.lower():
        return elongation_fixed

    # No rule matched - leave unchanged
    return original


def _fixed_form(token: str):
    """Rules that decide a token completely. Returns the result (an empty string means 'drop'), or None."""
    if token in FILLERS:
        return ""
    if token in LEAVE_ALONE:
        return token
    return standardize_spelling(token)


# Two-token contractions arrive from step 1 (apostrophes become spaces): "don't" -> "don" "t".
# Join them back into one token first, so the short-form list can expand them (dont -> do not).
_CONTRACTION_STEMS = {"don", "doesn", "didn", "isn", "wasn", "aren", "weren", "couldn", "can", "won",
                      "wouldn", "shouldn", "haven", "hasn", "hadn"}
# Other two-token short forms ("thank u" -> thank you)
PAIR_FIXES = {("thank", "u"): "thank you", ("thank", "yu"): "thank you", ("thank", "uu"): "thank you",
              ("thanks", "u"): "thank you", ("thank", "you"): "thank you"}


# Malay words that are doubled: 'sama ii' / 'sama2' -> 'sama sama' (no hyphen).
# ONLY these words are joined, so English like 'very very good' stays as it is. Add more words here.
MALAY_REPEAT_WORDS = set("""
sama diam kira lama baik cepat laju lambat perlahan murah mahal kecil besar banyak jalan makan hari
tahun orang kawan budak rumah buah kali lain pagi malam minggu bulan ramai cuba tunggu suka kejap
lalu tiba sikit hampir lurus senang susah kosong penuh jauh dekat tinggi rendah panas sejuk baru
lagi mana siapa apa kuat lemah kura cantik bagus tengah balik habis ikut sebut tanya pasang
""".split())


def merge_tokens(tokens):
    """['i', 'don', 't', 'like', 'thank', 'u'] -> ['i', 'dont', 'like', 'thank you']"""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt is not None and t.lower() in _CONTRACTION_STEMS and nxt.lower() == "t":
            out.append(t.lower() + "t")
            i += 2
        elif nxt is not None and t.lower() in MALAY_REPEAT_WORDS and nxt.lower() in ("ii", t.lower()):
            out.append(t.lower() + " " + t.lower())          # 'sama ii' (typed for sama2) -> 'sama sama'
            i += 2
        elif nxt is not None and (t.lower(), nxt.lower()) in PAIR_FIXES:
            out.append(PAIR_FIXES[(t.lower(), nxt.lower())])
            i += 2
        else:
            out.append(t)
            i += 1
    return out


def normalize_text(text: str) -> str:
    """Normalize a whole (already-tokenized, whitespace-joined) string."""
    tokens = merge_tokens(text.split())
    return " ".join(t for t in (normalize_token(x) for x in tokens) if t)   # fillers (la, lah) come back empty


if __name__ == "__main__":
    samples = [
        "broblom with my sim, netwerk keep dropping",
        "baddddd conection, custome servis so sloww",
        "mahal2 sgt pkej ni, tak leh guna langsung",
    ]
    for s in samples:
        print(f"IN : {s}")
        print(f"OUT: {normalize_text(s)}\n")