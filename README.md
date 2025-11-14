squidgame_sentiment/
├── app.py                → Streamlit web app (frontend)
├── notebooks/
│   └── SquidGame_Pipeline.ipynb  → E2E training + inference notebook
├── src/                  → Modular Python codebase
│   ├── config.py         → Centralized paths/config
│   ├── logger.py         → Custom logger (DD/MM/YYYY format)
│   ├── exceptions.py     → Custom exception classes
│   ├── utils.py          → Helper functions (text cleaning, weak labeling)
│   ├── data_prep.py      → Data loading & preparation
│   ├── eda.py            → Exploratory Data Analysis
│   └── modeling.py       → Model training & prediction
├── data/
│   └── tweets_v8.csv     → Input dataset
├── models/               → Saved ML artifacts (TF-IDF vectorizer + model)
├── eda/
│   ├── eda_report.json   → EDA summary
│   └── charts/           → Distribution plots
└── logs/
    └── app.log           → Runtime logs (with timestamps)

Modular pipeline with exception handling, DD/MM/YYYY logs, EDA, TF-IDF + Logistic Regression model, and Streamlit front-end.
