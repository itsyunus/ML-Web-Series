
from .logger import get_logger
from .exceptions import DataError
from .utils import find_text_and_label_cols, weak_sentiment_label, clean_text
from . import config
import pandas as pd

logger = get_logger("data_prep")

def load_raw(path=None) -> pd.DataFrame:
    try:
        p = config.DATA_PATH if path is None else path
        df = pd.read_csv(p, encoding="utf-8", engine="python", on_bad_lines="skip")
        logger.info(f"Loaded data: {df.shape}")
        return df
    except Exception as e:
        logger.exception("Failed to load CSV")
        raise DataError(str(e))

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    text_col, label_col = find_text_and_label_cols(df)
    if text_col is None:
        raise DataError("Could not find a text column.")
    if label_col is None:
        logger.info("No label column found. Creating weak labels from text/emoji rules.")
        df["__weak_label"] = df[text_col].map(weak_sentiment_label)
        df = df.dropna(subset=["__weak_label"]).copy()
        df["__weak_label"] = df["__weak_label"].astype(int)
        label_col = "__weak_label"
    df = df[[text_col, label_col]].dropna().copy()
    df["clean_text"] = df[text_col].map(clean_text)
    df = df[df["clean_text"].str.len() >= 3]
    df = df.drop_duplicates(subset=["clean_text"])
    logger.info(f"Prepared data: {df.shape} | text='{text_col}' label='{label_col}'")
    return df.rename(columns={label_col:"label"})
