
from .logger import get_logger
from .exceptions import DataError
from . import config
import pandas as pd, json, matplotlib.pyplot as plt

logger = get_logger("eda")

def run_eda(df: pd.DataFrame):
    try:
        report = {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "label_balance": df["label"].value_counts(dropna=False).to_dict(),
            "avg_len": float(df["clean_text"].str.len().mean() or 0.0),
            "top_words_sample": df["clean_text"].str.lower().str.split().explode().value_counts().head(20).to_dict(),
        }
        config.REPORT_PATH.write_text(json.dumps(report, indent=2))
        logger.info("EDA report saved at %s", str(config.REPORT_PATH))

        plt.figure()
        df["label"].value_counts().sort_index().plot(kind="bar")
        plt.title("Label Distribution")
        plt.xlabel("label")
        plt.ylabel("count")
        plt.tight_layout()
        (config.CHART_DIR / "label_distribution.png").parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(config.CHART_DIR / "label_distribution.png")
        plt.close()

        plt.figure()
        df["clean_text"].str.len().plot(kind="hist", bins=30)
        plt.title("Tweet Length Distribution")
        plt.xlabel("length (chars)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(config.CHART_DIR / "length_hist.png")
        plt.close()

        return report
    except Exception as e:
        logger.exception("EDA failed")
        raise DataError(str(e))
