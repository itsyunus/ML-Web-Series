
import re
import pandas as pd

TEXT_CANDIDATES = ["text","tweet","review","content","message","body"]
LABEL_CANDIDATES = ["label","sentiment","target","polarity","class","rating"]

def find_text_and_label_cols(df: pd.DataFrame):
    text_col = None
    for c in df.columns:
        if c.lower() in TEXT_CANDIDATES:
            text_col = c
            break
    if text_col is None:
        for c in df.columns:
            if df[c].dtype == "object":
                text_col = c
                break

    label_col = None
    for c in df.columns:
        if c.lower() in LABEL_CANDIDATES:
            label_col = c
            break
    return text_col, label_col

EMOJI_POS = set("🙂😊😀😃😄😆😍🥰😘👍🔥✨🤩👏😁😺💯🎉🤗🙌")
EMOJI_NEG = set("😞😟😠😡🤬😢😭👎💔🙁😣😖😫😩😤😔")
POS_WORDS = set("good great amazing awesome love superb fantastic brilliant outstanding fun nice happy wow masterpiece recommend solid engaging liked enjoyable emotional uplifting best intense thrilling mustwatch".split())
NEG_WORDS = set("bad boring awful terrible hate worst disappointing waste dull poor slow messy cringe annoying not_recommend flop weak bland drag worst".split())

def weak_sentiment_label(text: str) -> int:
    """Return 1 for positive, 0 for negative, based on simple rules.
    If neutral/unknown, return None to be filtered later.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.lower()
    pos_score = 0
    neg_score = 0
    for ch in t:
        if ch in EMOJI_POS: pos_score += 1
        if ch in EMOJI_NEG: neg_score += 1
    tokens = re.findall(r"[a-zA-Z']+", t)
    for tok in tokens:
        if tok in POS_WORDS: pos_score += 1
        if tok in NEG_WORDS: neg_score += 1
    if "#squidgame" in t and "love" in t: pos_score += 1
    if "#squidgame" in t and ("hate" in t or "worst" in t): neg_score += 1
    if pos_score == neg_score == 0:
        return None
    return 1 if pos_score >= neg_score else 0

def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = re.sub(r"http\S+"," ", s)
    s = re.sub(r"@\w+"," ", s)
    s = re.sub(r"#"," ", s)
    s = re.sub(r"\s+"," ", s)
    return s.strip()
