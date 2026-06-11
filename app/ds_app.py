import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent / 'src'))

import pickle
import numpy as np
import pandas as pd
import torch
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from preprocess import transform
from model import MLP
from evaluate import plot_results

ROOT            = pathlib.Path(__file__).parent.parent
CHECKPOINT_PATH = ROOT / 'models' / 'checkpoint.pkl'
GIF_PATH        = pathlib.Path(__file__).parent / 'titanic-sinking.gif'

st.set_page_config(page_title='Titanic Survival — Inference', layout='wide')

# ── Header ─────────────────────────────────────────────────────────────────────
st.title('🚢 Titanic Survival Prediction')

if GIF_PATH.exists():
    _, col, _ = st.columns([1, 2, 1])
    col.image(str(GIF_PATH), use_container_width=True)

st.markdown(
    'A PyTorch MLP trained on the Titanic dataset. '
    'Features include passenger class, title, age, family size, fare, cabin presence, and embarkation port.'
)

st.divider()

# ── Step 1: Upload dataset ─────────────────────────────────────────────────────
st.subheader('Step 1 — Provide a Dataset')
st.markdown('Upload any Titanic-format CSV. If none is uploaded, the default `data/train.csv` is used.')

uploaded = st.file_uploader('Upload a CSV file', type='csv')

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success(f'Loaded uploaded file — {len(df)} rows.')
else:
    default = ROOT / 'data' / 'train.csv'
    if default.exists():
        df = pd.read_csv(default)
        st.info(f'No file uploaded — using default `data/train.csv` ({len(df)} rows).')
    else:
        st.warning('No dataset found. Please upload a CSV above.')
        st.stop()

st.divider()

# ── Step 2: Load model ─────────────────────────────────────────────────────────
st.subheader('Step 2 — Load Trained Model')
st.markdown('Upload a trained checkpoint `.pkl` file produced by `src/train.py`, or leave blank to use the default.')

uploaded_ckpt = st.file_uploader('Upload checkpoint (.pkl)', type='pkl')

try:
    source = uploaded_ckpt if uploaded_ckpt is not None else open(CHECKPOINT_PATH, 'rb')
    ckpt = pickle.load(source)
    model = MLP()
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    label = 'uploaded checkpoint' if uploaded_ckpt is not None else '`models/checkpoint.pkl`'
    if uploaded_ckpt is not None:
        st.success(f'Checkpoint uploaded and loaded successfully.')
    else:
        st.info('No file uploaded — using default `models/checkpoint.pkl`.')
except FileNotFoundError:
    st.error('Checkpoint not found. Run `python src/train.py` first.')
    st.stop()

st.divider()

# ── Step 3: Run inference ──────────────────────────────────────────────────────
st.subheader('Step 3 — Run Inference')
st.markdown('Results load automatically. Click **Re-run** after uploading a new file.')

st.button('▶ Re-run Inference', type='primary')

if True:
    X = transform(df, ckpt['params'])
    with torch.no_grad():
        logits = model(torch.tensor(X)).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    LABEL = {0: 'Died', 1: 'Survived'}

    # Metrics
    if 'Survived' in df.columns:
        y = df['Survived'].values

        st.subheader('Performance Metrics')
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Accuracy',  f'{accuracy_score(y, preds):.3f}')
        c2.metric('Precision', f'{precision_score(y, preds):.3f}')
        c3.metric('Recall',    f'{recall_score(y, preds):.3f}')
        c4.metric('F1 Score',  f'{f1_score(y, preds):.3f}')
        c5.metric('ROC-AUC',   f'{roc_auc_score(y, probs):.3f}')

        st.subheader('Evaluation Plots')
        st.pyplot(plot_results(y, probs))

    else:
        st.info('No `Survived` column found — showing predictions only.')

    # Predictions table
    st.subheader('Predictions')
    out = pd.DataFrame({
        'Name':       df['Name'] if 'Name' in df.columns else df.index,
        'Pclass':     df['Pclass'],
        'Sex':        df['Sex'],
        'Age':        df['Age'],
        'Actual':     df['Survived'].map(LABEL) if 'Survived' in df.columns else '—',
        'Prediction': pd.Series(preds).map(LABEL),
        'P(Survive)': probs.round(3),
    })
    st.dataframe(out, use_container_width=True)
