import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

import copy, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from preprocess import fit_transform, transform
from model import MLP

DATA_DIR   = pathlib.Path(__file__).parent.parent / 'data'
DATA_PATH  = DATA_DIR / 'train.csv'
SAVE_PATH  = pathlib.Path(__file__).parent.parent / 'models' / 'checkpoint.pkl'
SEED       = 42
EPOCHS     = 200
BATCH_SIZE = 32
LR         = 1e-3
PATIENCE   = 15


def fetch_data():
    if DATA_PATH.exists():
        print(f'✓ Using cached {DATA_PATH}')
        return
    DATA_DIR.mkdir(exist_ok=True)
    try:
        import kaggle, zipfile
        kaggle.api.authenticate()
        kaggle.api.competition_download_file('titanic', 'train.csv', path=DATA_DIR)
        zip_path = DATA_DIR / 'train.csv.zip'
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(DATA_DIR)
            zip_path.unlink()
        print('✓ Downloaded train.csv from Kaggle API.')
    except Exception as e:
        print(f'Kaggle unavailable ({e}) — falling back to GitHub mirror.')
        import urllib.request
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
            DATA_PATH
        )
        print('✓ Downloaded from GitHub mirror.')


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    import time
    t_start = time.time()

    # --- Data ---
    fetch_data()
    df = pd.read_csv(DATA_PATH)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df['Survived']
    )

    X_train, y_train, params = fit_transform(train_df)
    X_val                    = transform(val_df, params)
    y_val                    = val_df['Survived'].values.astype(np.float32)

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t   = torch.tensor(X_val)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE, shuffle=True
    )

    # --- Model ---
    model     = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-5
    )

    # Weighted loss to handle class imbalance (62% died / 38% survived)
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Training loop with early stopping ---
    best_auc        = 0.0
    best_state      = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).numpy()
        val_probs = 1 / (1 + np.exp(-val_logits))   # sigmoid
        val_preds = (val_probs >= 0.5).astype(int)
        val_acc   = (val_preds == y_val).mean()
        val_auc   = roc_auc_score(y_val, val_probs)

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss/len(train_loader):.4f}'
                  f'  val_acc={val_acc:.4f}  val_auc={val_auc:.4f}  lr={lr_now:.2e}')

        if epochs_no_improve >= PATIENCE:
            print(f'\nEarly stopping at epoch {epoch} (best val_auc={best_auc:.4f})')
            best_epoch = epoch - PATIENCE
            break
    else:
        best_epoch = EPOCHS

    # --- Restore best model and print final metrics ---
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t).numpy()
    val_probs = 1 / (1 + np.exp(-val_logits))

    from evaluate import print_metrics
    print_metrics(y_val, val_probs)

    # --- Save ---
    SAVE_PATH.parent.mkdir(exist_ok=True)
    val_preds = (val_probs >= 0.5).astype(int)
    val_metrics = {
        'accuracy':  float(accuracy_score(y_val, val_preds)),
        'precision': float(precision_score(y_val, val_preds)),
        'recall':    float(recall_score(y_val, val_preds)),
        'f1':        float(f1_score(y_val, val_preds)),
        'roc_auc':   float(roc_auc_score(y_val, val_probs)),
    }
    val_data = {'y_true': y_val, 'y_probs': val_probs}

    with open(SAVE_PATH, 'wb') as f:
        pickle.dump({
            'model_state':    model.state_dict(),
            'params':         params,
            'val_metrics':    val_metrics,
            'val_data':       val_data,
            'train_metadata': {
                'epochs':       best_epoch,
                'batch_size':   BATCH_SIZE,
                'train_time_s': round(time.time() - t_start, 1),
            },
        }, f)
    print(f'\nCheckpoint saved → {SAVE_PATH}')


if __name__ == '__main__':
    train()
