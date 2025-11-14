
import streamlit as st
import pandas as pd
from src import config
from src.modeling import predict
from src.logger import get_logger

logger = get_logger("app")

st.set_page_config(page_title="SquidGame Tweet Sentiment", layout="centered")
st.title("🐙 SquidGame Tweet Sentiment")
st.caption("Log date format: DD/MM/YYYY. Model: TF-IDF + Logistic Regression.")

with st.expander("Batch predict (CSV with a 'text' column)"):
    up = st.file_uploader("Upload CSV", type=["csv"], key="csv")
    if up:
        df = pd.read_csv(up)
        text_col = "text" if "text" in df.columns else df.columns[0]
        preds, probs = predict(df[text_col].astype(str).tolist())
        out = df.copy()
        out["pred_label"] = preds
        if probs is not None:
            out["pred_prob_pos"] = probs
        st.dataframe(out.head(50))
        st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"), "predictions.csv")

st.subheader("Single text")
txt = st.text_area("Enter a tweet/review", height=120, placeholder="That marble episode of #SquidGame ruined me 😭😭😭")
if st.button("Predict"):
    try:
        preds, probs = predict([txt])
        label = int(preds[0])
        prob = float(probs[0]) if probs is not None else None
        st.write(f"**Prediction:** {'Positive 👍' if label==1 else 'Negative 👎'}")
        if prob is not None:
            st.write(f"**Confidence (P=positive):** {prob:.3f}")
        logger.info(f"Prediction made | label={label} prob={prob}")
    except Exception as e:
        st.error(f"Error: {e}")
        logger.exception("App prediction failed")
