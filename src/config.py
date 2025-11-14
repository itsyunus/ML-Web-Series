
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "tweets_v8.csv"
MODEL_DIR = BASE_DIR / "models"
EDA_DIR = BASE_DIR / "eda"
MODEL_PATH = MODEL_DIR / "model.pkl"
VECT_PATH = MODEL_DIR / "vectorizer.pkl"
REPORT_PATH = EDA_DIR / "eda_report.json"
CHART_DIR = EDA_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
