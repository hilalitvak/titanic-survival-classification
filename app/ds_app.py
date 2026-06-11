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
REQUIRED_COLS   = {'Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'}

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

# ── Step 1: Load model + show validation results ───────────────────────────────
st.subheader('Step 1 — Model Evaluation')
checkpoint_input = st.text_input('Checkpoint path', value=str(CHECKPOINT_PATH))
CHECKPOINT_PATH  = pathlib.Path(checkpoint_input)
st.markdown('The trained model is loaded from the path above and evaluated on the held-out validation set.')

try:
    with open(CHECKPOINT_PATH, 'rb') as f:
        ckpt = pickle.load(f)
    model = MLP()
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    st.success('Model loaded successfully from `models/checkpoint.pkl`.')
except FileNotFoundError:
    st.error('Checkpoint not found. Run `python src/train.py` first.')
    st.stop()

m = ckpt['val_metrics']
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Accuracy',  f'{m["accuracy"]:.3f}')
c2.metric('Precision', f'{m["precision"]:.3f}')
c3.metric('Recall',    f'{m["recall"]:.3f}')
c4.metric('F1 Score',  f'{m["f1"]:.3f}')
c5.metric('ROC-AUC',   f'{m["roc_auc"]:.3f}')

vd = ckpt['val_data']
st.pyplot(plot_results(vd['y_true'], vd['y_probs']))

st.divider()

# ── Step 2: Upload dataset ─────────────────────────────────────────────────────
st.subheader('Step 2 — Provide a Dataset')
st.markdown('Upload a Titanic-format CSV to run inference on new passengers.')

uploaded = st.file_uploader('Upload a CSV file', type='csv')

if uploaded is not None:
    df = pd.read_csv(uploaded)
    if df.empty:
        st.error('Uploaded file is empty.')
        st.stop()
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f'Uploaded file is missing required columns: `{", ".join(sorted(missing))}`')
        st.stop()
    st.success(f'Dataset uploaded successfully — {len(df)} rows.')
else:
    st.info('No file uploaded yet — upload a CSV above to run inference.')
    st.stop()

st.divider()

# ── Step 3: Run inference ──────────────────────────────────────────────────────
st.subheader('Step 3 — Run Inference')

X = transform(df, ckpt['params'])
with torch.no_grad():
    logits = model(torch.tensor(X)).numpy()
probs = 1 / (1 + np.exp(-logits))
preds = (probs >= 0.5).astype(int)

LABEL = {0: 'Died', 1: 'Survived'}

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
