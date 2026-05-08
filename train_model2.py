"""
train_model.py — Trains two calibrated RF models (reply + ghosted)
Run once: python train_model.py
Outputs: rf_reply_model.pkl, rf_ghost_model.pkl, model_metrics.pkl

ROOT CAUSE OF 100% ACCURACY BUG:
  cv='prefit' was removed in sklearn 1.6+. When train_model2.py used cv=5,
  sklearn refitted the calibrator 5 times using the test set as both
  calibration data AND evaluation data → pure leakage → 100% accuracy.

FIX: Proper 3-way split so no set is ever used twice:
  70% train  → fit the Random Forest
  15% val    → fit the CalibratedClassifierCV (calibration only)
  15% test   → evaluate final metrics (never seen by either step above)
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, brier_score_loss
)
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('ghosting_dataset5.csv')
print(f"Loaded {len(df)} rows")

df['effort_score']     = df['last_message_length'] + (df['emoji_count'] * 2) + (df['question_asked'] * 5)
df['delay']            = df['response_time_gap'].apply(lambda x: 0 if x < 6 else 1 if x < 24 else 2)
df['is_dry']           = (df['message_tone'] == 'dry').astype(int)
df['is_long_gap']      = (df['response_time_gap'] > 24).astype(int)
df['engagement_score'] = df['reply_ratio'] * df['conversation_length']
df['ghost_risk_combo'] = ((df['response_time_gap'] > 24) & (df['reply_ratio'] < 0.4)).astype(int)
df['seen_delay']       = ((df['seen_ignored'] == 1) & (df['response_time_gap'] > 12)).astype(int)
df['initiator_flag']   = (df['initiator'] == 'me').astype(int)
df['inconsistency']    = (abs(df['response_time_gap'] - df['avg_response_time']) > 20).astype(int)
df['decay_score']      = (df['conversation_length'] / 200).clip(0, 1)
df['effort_mismatch']  = ((df['last_message_length'] > 20) & (df['reply_ratio'] < 0.3)).astype(int)

NUM_FEATURES = [
    'last_message_length', 'response_time_gap', 'conversation_length',
    'reply_ratio', 'avg_response_time', 'emoji_count', 'question_asked',
    'seen_ignored', 'past_ghosting_history', 'effort_score', 'delay',
    'is_dry', 'is_long_gap', 'engagement_score', 'ghost_risk_combo',
    'seen_delay', 'initiator_flag', 'inconsistency', 'decay_score', 'effort_mismatch'
]
CAT_FEATURES = ['initiator', 'message_tone', 'time_of_day', 'user_type']

def make_preprocessor():
    num_t = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat_t = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer(
        [('num', num_t, NUM_FEATURES), ('cat', cat_t, CAT_FEATURES)],
        remainder='drop'
    )

def train_target(df, target_col, label):
    print(f"\n{'='*50}\nTraining: {label}\n{'='*50}")
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[target_col]

    # Step 1: hold out 15% as final test (never touched during training or calibration)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    # Step 2: split remaining into train (70%) + calibration val (15%)
    val_frac = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=42, stratify=y_trainval
    )
    print(f"Split → train:{len(X_train)}  val:{len(X_val)}  test:{len(X_test)}")

    # Step 3: fit RF on train only
    rf = Pipeline([
        ('preprocessor', make_preprocessor()),
        ('clf', RandomForestClassifier(
            n_estimators=400, max_depth=20,
            class_weight='balanced', random_state=42, n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)

    # Step 4: fit calibrator on val only (cv=3 does 3-fold CV within val)
    cal = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
    cal.fit(X_val, y_val)

    # Step 5: evaluate on clean test set
    y_pred  = cal.predict(X_test)
    y_proba = cal.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    cm    = confusion_matrix(y_test, y_pred)
    rep   = classification_report(y_test, y_pred)

    print(f"Accuracy : {acc:.4f}  |  F1: {f1:.4f}  |  AUC: {auc:.4f}  |  Brier: {brier:.4f}")
    print(rep)

    return cal, {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1_score': f1, 'roc_auc': auc, 'brier': brier,
        'confusion_matrix': cm.tolist(),
        'classification_report': rep,
        'train_size': len(X_train),
        'val_size':   len(X_val),
        'test_size':  len(X_test),
        'y_test':     y_test.tolist(),
        'y_pred_prob': y_proba.tolist(),
        'label': label
    }

reply_model, reply_metrics = train_target(df, 'reply',   'REPLY MODEL')
ghost_model, ghost_metrics = train_target(df, 'ghosted', 'GHOST MODEL')

joblib.dump(reply_model,  'rf_reply_model.pkl')
joblib.dump(ghost_model,  'rf_ghost_model.pkl')
joblib.dump({'reply': reply_metrics, 'ghosted': ghost_metrics}, 'model_metrics.pkl')

print("\n✅ Saved: rf_reply_model.pkl  rf_ghost_model.pkl  model_metrics.pkl")
print(f"\nTrue metrics (no leakage):")
print(f"  Reply   — Acc: {reply_metrics['accuracy']:.1%}  F1: {reply_metrics['f1_score']:.3f}  AUC: {reply_metrics['roc_auc']:.3f}")
print(f"  Ghosted — Acc: {ghost_metrics['accuracy']:.1%}  F1: {ghost_metrics['f1_score']:.3f}  AUC: {ghost_metrics['roc_auc']:.3f}")