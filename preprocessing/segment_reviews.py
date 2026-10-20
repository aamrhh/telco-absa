"""
segment_reviews.py - v4: cut a RAW review into one part per aspect (v3 + the fixes from the 300-part check).

Order (the plan we agreed):
  1. Work on the RAW review (it still has . , ! ? and new lines). Lowercase it.
  2. Cut at EVERY marker:  . ! ? ; new-line   ,   but tapi however namun ...   and dan serta plus while
     -> small pieces.
  3. Find the aspect word(s) in each piece (same lists as detect_aspects.py; app/aplikasi = app aspect).
  4. Merge pieces:
        - a piece with NO aspect word joins the piece before it (or the next one, if it is the first)
        - neighbours with the SAME aspect join each other   ("line slow, speed bad" = one part)
        - a piece of fewer than --min-words words joins its neighbour
        - a piece that holds 2 aspects with no marker between them stays together (aspect "a|b")
  5. Same aspect on both sides of a contrast word ("line fast but unstable") -> one part, maybe_mixed = 1.

v4 CHANGES (from your one-by-one check, numbers from count_v4_questions.py):
  keywords  store / complain / complaint / update / loading / load / pay-bill / network are no longer plain keywords:
            store (Play Store, App Store) ignored; shop 'store' only with staff / branch / a telco name
            complaint counts only as 'lodge / file / make a complaint', 'complain to'
            update counts as app unless a network word is in the same part (antenna, tower, line ...)
            loading = service if an internet word is in the part and no app word, otherwise app
            pay / bill / payment = app only with cannot / error / failed; price talk = none
            network ignored in 'change / switch network' and 'sign in to network'
            crash = app, but not when a game / streaming word is in the part
            helpline / help line = customer service; livechat / online chat = customer service
            internet / wifi + a problem word within 3 words = service reliability
            red bar / no bars = service reliability; easy to use / user friendly / frenly / login = app
  cutting   (A) a piece with no aspect but an opinion, ending with . ! ? ; and 3+ words, stays alone (aspect none)
            (B) 'and' does not cut when the left piece has no opinion yet ("my mobile data | and wifi ... not working")
            (C) a piece that starts with if / kalau / jika joins the next piece
            (F) a full stop between two digits (3.5mb) is not a cut

NOT done yet (needs an opinion-word list, next step): "the customer service and the coverage are awful"
(one opinion shared by two aspects). Those parts show up as short parts with no opinion of their own.

Nothing in your real data changes. It only writes a new file and prints examples to read.

Run from the project root:
    python3 preprocessing/segment_reviews.py
    python3 preprocessing/segment_reviews.py --show 40        print 40 random cuts
    python3 preprocessing/segment_reviews.py --no-app         'app' does NOT count as an aspect word
    python3 preprocessing/segment_reviews.py --id 7677        show one review

Reads:  4_dictionary_mapped_reviews.csv  (columns review_id, review_text)
Writes: review_segments_raw.csv   review_id, segment_id, segment_text, aspect, n_aspects_in_review, cut_reason, maybe_mixed
"""
import argparse, re, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_aspects import load_keywords, compile_patterns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "processed"
APP = re.compile(r"\b(?:app|apps|application|applications|aplikasi)\b")
CONTRAST = r"\b(?:but|tapi|tetapi|however|namun|walaupun|though|although|cuma|tp)\b"
JOIN = r"\b(?:and|dan|serta|plus|while|manakala)\b"
MARK = re.compile(rf"(?P<end>[!?;\n]+|(?<!\d)\.+|\.+(?!\d))|(?P<comma>,)|(?P<contrast>{CONTRAST})|(?P<join>{JOIN})", re.I)
KIND = {"end": "sentence end", "contrast": "contrast word", "join": "joining word", "comma": "comma"}
RANK = ["end", "contrast", "join", "comma"]            # if several markers sit together, keep the strongest
TRIM = " ,.;!?-\n\t"


# ---- v4: words taken OUT of the keyword file (handled by the rules below instead) -------------------------------
DROP_KEYWORDS = {"store", "complaint", "complaints", "complain", "complained", "update", "updates", "updated",
                 "updating", "loading", "load", "network"}

STORE_MASK = re.compile(r"\b(?:google\s+)?(?:play\s*store|app\s*store|playstore|appstore)\b")
HELPLINE = re.compile(r"\bhelp[\s-]?line\b")
SWITCH_NET = re.compile(r"\b(?:change|changed|changing|switch|switched|switching|move|moved|moving|tukar|pindah)\b(?:\s+\w+){0,2}\s+network\b")
SIGNIN_NET = re.compile(r"\b(?:sign|log)\s*in\s+to\s+(?:the\s+)?network\b|\bnetwork\s+(?:sign|log)\s*in\b")
NETWORK = re.compile(r"\bnetwork\b|\bnetwook\b")
CRASH = re.compile(r"\bcrash(?:es|ed|ing)?\b")
GAME = re.compile(r"\b(?:game|games|gaming|ml|mobile\s+legends?|pubg|roblox|tiktok|youtube|netflix|stream\w*)\b")
UPDATE = re.compile(r"\b(?:update|updates|updated|updating|kemaskini|kemas\s+kini)\b")
NET_UPDATE = re.compile(r"\b(?:antenna|tower|line|lines|coverage|network|signal|rangkaian|menara|isyarat)\b")
LOADING = re.compile(r"\b(?:loading|load|loaded)\b")
NETWORKISH = re.compile(r"\b(?:internet|data|signal|wifi|wi-fi|4g|5g|line|coverage|network|connection|talian|rangkaian)\b")
PAYW = re.compile(r"\b(?:pay|payment|payments|bill|bills|bayar|bayaran)\b")
FAILW = re.compile(r"\b(?:cannot|can't|cant|can\s+not|can\s+t|unable|error|errors|fail|failed|failure|gagal|tak\s+boleh|tidak\s+boleh|stuck|problem)\b")
COMPLAINT_CS = re.compile(r"\b(?:lodge|lodged|file|filed|make|made|submit|submitted|raise|raised|send|sent)\s+(?:a\s+|an\s+|the\s+|my\s+)?(?:formal\s+)?complaints?\b"
                          r"|\bcomplain(?:ed|ing|s)?\s+(?:to|at|with)\b|\bcomplaints?\s+(?:to|at)\b")
STORE_SHOP = re.compile(r"\b(?:celcom|digi|celcomdigi|maxis|umobile|u\s+mobile)\s+(?:store|shop)\b"
                        r"|\b(?:store|branch|outlet)\s+(?:staff|counter)\b|\b(?:staff|counter)\s+(?:at|in)\s+(?:the\s+)?store\b")
INTERNET_PROBLEM = re.compile(r"\b(?:internet|wifi|wi-fi|net)\b(?:\W+\w+){0,3}?\W+(?:slow|lembab|lembap|lambat|siput|putus|poor|teruk|lag|lagging|down|bad|worst|unstable|terrible|lemah)\b"
                              r"|\b(?:slow|lembab|lembap|lambat|siput|putus|poor|teruk|lag|lagging|down|bad|worst|unstable|terrible|lemah)\b(?:\W+\w+){0,3}?\W+(?:internet|wifi|wi-fi)\b")
BARS = re.compile(r"\b(?:red\s+bar|bar\s+red|bar\s+merah|signal\s+bars?|no\s+bars?|bar\s+signal|one\s+bar|1\s+bar)\b")
CS_EXTRA = re.compile(r"\b(?:livechat|live\s+chat|online\s+chat|chat\s+support|chat\s+agent|helpdesk|hotline)\b")
APP_EXTRA = re.compile(r"\b(?:user[\s-]?friendly|frenly|easy\s+to\s+(?:use|navigate)|easy\s+use|simple\s+to\s+use|simple\s+app|"
                       r"hard\s+to\s+use|difficult\s+to\s+use|mudah\s+(?:guna|digunakan)|senang\s+guna|susah\s+guna|"
                       r"logins?|verification|verify|past\s+bills?|(?:cannot|can't|cant|unable\s+to|can\s+not)\s+access)\b")
OPINION = re.compile(r"\b(?:good|bad|great|best|worst|terrible|awful|love|hate|nice|excellent|useless|disappoint\w*|recommend\w*|"
                     r"satisf\w*|happy|thank\w*|bagus|teruk|puas|hampa|kecewa|mantap|hancur|lembab|lembap|slow|fast|laju|lambat|"
                     r"poor|horrible|amazing|worse|waste|sucks?|scam|fix|please|tolong|sampah|smooth|stable|unstable|problem|error|"
                     r"cannot|can't|cant|crash\w*|fail\w*|expensive|mahal|cheap|murah|okay|ok|fine|annoying|frustrat\w*|stupid|"
                     r"bodoh|trash|pathetic|ridiculous|improve\w*|work|works|working|simple|easy|useful|helpful|unhelpful|lousy|"
                     r"worth|issue|issues|tak\s+boleh|susah)\b")
# rule (A) only: words that show a GENERAL opinion (not "fail", "fix", "please", "problem": those continue a complaint)
GENERAL_OPINION = re.compile(r"\b(?:good|bad|great|best|worst|terrible|awful|love|hate|nice|excellent|useless|disappoint\w*|"
                             r"recommend\w*|satisf\w*|happy|thank\w*|bagus|teruk|puas|hampa|kecewa|mantap|horrible|amazing|"
                             r"waste|sucks?|scam|sampah|pathetic|ridiculous|lousy|worth|okay|ok|fine|annoying|frustrat\w*|"
                             r"stupid|bodoh|trash|improve\w*|simple|easy|works|smooth)\b")
CONTINUES = re.compile(r"^\s*(?:also|along|and|but|then|because|so|plus|as\s+well|too|still|tried|try|even|until|which|that|"
                       r"is|are|was|were|to|of|for|with|about|making|who|when|while|after|before|if|it)\b")
IF_START = re.compile(r"^\s*(?:if|kalau|kalo|jika|sekiranya)\b")


def has_opinion(text):
    return OPINION.search(text) is not None


def filter_keywords(by_aspect):
    """Drop the words handled by rules; the keyword FILE itself is not changed."""
    return {a: [w for w in ws if w not in DROP_KEYWORDS] for a, ws in by_aspect.items()}


def aspects_in(text, patterns, app_counts):
    t = STORE_MASK.sub(" ", text)
    t = HELPLINE.sub(" hotline ", t)
    t = SWITCH_NET.sub(" ", t)
    t = SIGNIN_NET.sub(" ", t)
    if CRASH.search(t) and GAME.search(t):
        t = CRASH.sub(" ", t)
    found = {a for a, pat in patterns.items() if pat.search(t)}
    if app_counts and APP.search(t):
        found.add("app_performance")
    # service reliability
    if NETWORK.search(t) or INTERNET_PROBLEM.search(t) or BARS.search(t):
        found.add("service_reliability")
    # customer service
    if COMPLAINT_CS.search(t) or STORE_SHOP.search(t) or CS_EXTRA.search(t):
        found.add("customer_service")
    # app
    if APP_EXTRA.search(t):
        found.add("app_performance")
    if UPDATE.search(t) and not NET_UPDATE.search(t):
        found.add("app_performance")
    if LOADING.search(t):
        found.add("service_reliability" if NETWORKISH.search(t) and not APP.search(t) else "app_performance")
    if PAYW.search(t) and FAILW.search(t):
        found.add("app_performance")
    return found


def make_pieces(text):
    """Cut at every marker. Returns list of dict(start, end, kind_after, contrast_before)."""
    pieces, pos, pending = [], 0, set()
    for m in MARK.finditer(text):
        if text[pos:m.start()].strip(TRIM):
            pieces.append({"start": pos, "end": m.start(), "after": None, "contrast_before": False})
            if len(pieces) > 1:
                pieces[-2]["after"] = pending
                pieces[-1]["contrast_before"] = "contrast" in pending
            pending = set()
        pending.add(m.lastgroup)
        pos = m.end()
    if text[pos:].strip(TRIM):
        pieces.append({"start": pos, "end": len(text), "after": None, "contrast_before": False})
        if len(pieces) > 1:
            pieces[-2]["after"] = pending
            pieces[-1]["contrast_before"] = "contrast" in pending
    return pieces


def kind_of(after):
    for k in RANK:
        if after and k in after:
            return KIND[k]
    return None


def segment(text, patterns, app_counts=True, min_words=2):
    text = str(text).lower().replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    pieces = make_pieces(text)
    if not pieces:
        return [(text.strip(TRIM) or text.strip(), "none", "not cut", 0)], 0
    # (C) a piece that starts with if / kalau / jika joins the next piece
    # (B) 'and' does not cut when the left piece has no opinion yet
    i = 0
    while i < len(pieces) - 1:
        a, b = pieces[i], pieces[i + 1]
        left = text[a["start"]:a["end"]]
        join_only = kind_of(a["after"]) == "joining word"
        if IF_START.match(left) or (join_only and not has_opinion(left)):
            a["end"], a["after"] = b["end"], b["after"]
            del pieces[i + 1]
            continue
        i += 1
    for p in pieces:
        p["asp"] = aspects_in(text[p["start"]:p["end"]], patterns, app_counts)
        p["mixed"] = False
        piece_txt = text[p["start"]:p["end"]]
        # (A) no aspect word, but an opinion, 3+ words and a sentence end: it stays alone (aspect none)
        ends = (p["after"] and "end" in p["after"]) or re.match(r"\s*[.!?;]", text[p["end"]:]) is not None
        p["alone"] = (not p["asp"]) and bool(ends) and len(piece_txt.split()) >= 3 \
                      and GENERAL_OPINION.search(piece_txt) is not None and CONTINUES.match(piece_txt) is None

    # 1. pieces without an aspect word join the previous piece (or the next one, if first)
    merged, pend = [], None
    for p in pieces:
        if not p["asp"] and not p["alone"]:
            if merged:
                merged[-1]["end"], merged[-1]["after"] = p["end"], p["after"]
                merged[-1]["mixed"] |= p["contrast_before"]
            elif pend is None:
                pend = dict(p)
            else:
                pend["end"] = p["end"]
            continue
        if pend is not None:
            p["start"] = pend["start"]
            pend = None
        merged.append(p)
    if not merged:                                   # no aspect word anywhere
        return [(text.strip(TRIM), "none", "not cut", 0)], 0

    # 2. neighbours with the same aspect(s) join. A contrast word between them = maybe mixed
    def join_same(parts):
        out = [parts[0]]
        for p in parts[1:]:
            q = out[-1]
            if q["asp"] == p["asp"]:
                q["end"], q["after"] = p["end"], p["after"]
                q["mixed"] = q["mixed"] or p["contrast_before"] or p["mixed"]
            else:
                out.append(p)
        return out
    merged = join_same(merged)

    # 3. very short parts join a neighbour (their aspect is added: a|b)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, p in enumerate(merged):
            if len(text[p["start"]:p["end"]].split()) < min_words:
                j = i - 1 if i > 0 else i + 1
                lo, hi = min(i, j), max(i, j)
                a, b = merged[lo], merged[hi]
                a["end"], a["after"] = b["end"], b["after"]
                a["asp"] = a["asp"] | b["asp"]
                a["mixed"] = a["mixed"] or b["contrast_before"] or b["mixed"]
                a["short_merged"] = True
                del merged[hi]
                merged = join_same(merged)
                changed = True
                break

    n_rev = len(aspects_in(text, patterns, app_counts))
    out = []
    for i, p in enumerate(merged):
        asp = "|".join(sorted(p["asp"])) or "none"
        if len(merged) == 1:
            why = "not cut"
        elif i == len(merged) - 1:
            why = "last part"
        else:
            why = kind_of(p["after"]) or "cut"
        if p.get("short_merged"):
            why += " (short part merged)"
        out.append((re.sub(r"\s+", " ", text[p["start"]:p["end"]]).strip(TRIM), asp, why, int(bool(p["mixed"] and len(p["asp"]) == 1))))
    return [o for o in out if o[0]], n_rev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="4_dictionary_mapped_reviews.csv")
    ap.add_argument("--no-app", action="store_true", help="do not count app/aplikasi as an aspect word")
    ap.add_argument("--min-words", type=int, default=2, help="parts shorter than this join a neighbour")
    ap.add_argument("--show", type=int, default=25)
    ap.add_argument("--id", default=None, help="show only this review_id")
    args = ap.parse_args()

    patterns = compile_patterns(filter_keywords(load_keywords()))
    df = pd.read_csv(DATA / args.file, dtype={"review_id": str}, usecols=["review_id", "review_text"])
    df["review_text"] = df["review_text"].fillna("").astype(str)
    if args.id:
        df = df[df["review_id"] == args.id]
        for rid, t in zip(df["review_id"], df["review_text"]):
            print(f"[{rid}] RAW: {t}")
            for txt, asp, why, mixed in segment(t, patterns, not args.no_app, args.min_words)[0]:
                print(f"   ({asp}) {txt}   <- {why}{'  [maybe mixed]' if mixed else ''}")
        return

    rows = []
    for rid, t in zip(df["review_id"], df["review_text"]):
        parts, n = segment(t, patterns, not args.no_app, args.min_words)
        for i, (txt, asp, why, mixed) in enumerate(parts, 1):
            rows.append((rid, i, txt, asp, n, why, mixed))
    out = pd.DataFrame(rows, columns=["review_id", "segment_id", "segment_text", "aspect",
                                      "n_aspects_in_review", "cut_reason", "maybe_mixed"])
    out.to_csv(DATA / "review_segments_raw.csv", index=False)

    per = out.groupby("review_id").agg(n_asp=("n_aspects_in_review", "first"), n_seg=("segment_id", "max"))
    print(f"{len(per)} reviews -> {len(out)} parts   ({int((per.n_seg > 1).sum())} reviews were cut)")
    print("Reviews by number of aspects found:", per["n_asp"].value_counts().sort_index().to_dict())
    print("Parts by aspect:", out["aspect"].value_counts().to_dict())
    print("Parts that hold 2+ aspects (no marker between them, kept together):", int(out["aspect"].str.contains(r"\|").sum()))
    print("Parts flagged maybe mixed (one aspect, contrast word inside):", int(out["maybe_mixed"].sum()))
    print("Why parts were cut:", out["cut_reason"].value_counts().to_dict())
    multi = out[out.groupby("review_id")["segment_id"].transform("max") > 1]
    if len(multi):
        ids = multi["review_id"].drop_duplicates().sample(min(args.show, multi["review_id"].nunique()), random_state=1)
        print(f"\n--- {len(ids)} random reviews that were cut. READ THESE and say which look wrong ---")
        for rid in ids:
            g = out[out.review_id == rid]
            print(f"\n[{rid}]")
            for _, r in g.iterrows():
                print(f"   ({r.aspect}) {r.segment_text}   <- {r.cut_reason}")
    print("\nSaved review_segments_raw.csv")


if __name__ == "__main__":
    main()