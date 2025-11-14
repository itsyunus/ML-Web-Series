
from .logger import get_logger
from .exceptions import ModelError
from . import config
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

logger = get_logger("modeling")

def train_and_save(df: pd.DataFrame, random_state: int = 42):
    try:
        X = df["clean_text"].tolist()
        y = df["label"].astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        vect = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)
        Xtr = vect.fit_transform(X_train)
        Xte = vect.transform(X_test)

        clf = LogisticRegression(max_iter=200)
        clf.fit(Xtr, y_train)
        preds = clf.predict(Xte)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        config.MODEL_DIR.mkdir(exist_ok=True, parents=True)
        joblib.dump(clf, config.MODEL_PATH)
        joblib.dump(vect, config.VECT_PATH)

        logger.info(f"Saved model to {config.MODEL_PATH} | vectorizer to {config.VECT_PATH} | acc={acc:.4f}")
        return {"accuracy": float(acc), "report": report}
    except Exception as e:
        logger.exception("Training failed")
        raise ModelError(str(e))

def load_model():
    import joblib
    clf = joblib.load(config.MODEL_PATH)
    vect = joblib.load(config.VECT_PATH)
    return clf, vect

def predict(texts):
    try:
        clf, vect = load_model()
        X = vect.transform(texts)
        preds = clf.predict(X)
        probs = clf.predict_proba(X)[:,1] if hasattr(clf, "predict_proba") else None
        return preds, probs
    except Exception as e:
        logger.exception("Inference failed")
        raise ModelError(str(e))
