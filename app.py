import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, brier_score_loss,
                              confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="💀 Ghosting Predictor",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.main-title {
    font-family: 'Syne', sans-serif; font-size: 3.0em; font-weight: 700;
    color: #ff6b6b; /* Set text color to red */
    text-align: center; margin-bottom: 20px; letter-spacing: 0.5px;
}
.sub-title { text-align: center; color: #b2bec3; font-size: 1.2em; margin-top: 10px; margin-bottom: 20px; }

.verdict-card {
    border-radius: 20px; padding: 30px; text-align: center;
    margin: 20px auto; position: relative; overflow: hidden;
    max-width: 700px;
}
.verdict-high  { background: linear-gradient(135deg, #00b894, #00cec9); color: white; }
.verdict-mid   { background: linear-gradient(135deg, #fdcb6e, #e17055); color: white; }
.verdict-low   { background: linear-gradient(135deg, #d63031, #6c5ce7); color: white; }
.verdict-pct   { font-family: 'Syne', sans-serif; font-size: 3em; font-weight: 800; line-height: 1.2; }
.verdict-label { font-size: 1.2em; font-weight: 600; margin-top: 10px; opacity: 0.9; }
.verdict-quote { font-size: 1em; margin-top: 16px; font-style: italic; opacity: 0.85;
                 border-top: 1px solid rgba(255,255,255,0.2); padding-top: 16px; }

.diag-row { display: flex; gap: 20px; margin: 20px auto; flex-wrap: wrap; justify-content: center; }
.diag-card {
    flex: 1; min-width: 180px; max-width: 250px; border-radius: 15px; padding: 20px;
    text-align: center; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.1);
}
.diag-title { font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; color: #b2bec3; margin-bottom: 8px; }
.diag-val   { font-family: 'Syne', sans-serif; font-size: 1.5em; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.val-high   { color: #00b894; }
.val-mid    { color: #fdcb6e; }
.val-low    { color: #ff7675; }

.flag-item {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 16px; border-radius: 12px; margin: 8px auto;
    background: rgba(214, 48, 49, 0.15); border-left: 4px solid #d63031; font-size: 1em;
    max-width: 700px;
}
.green-flag { background: rgba(0, 184, 148, 0.15); border-left: 4px solid #00b894; }

.share-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 20px; padding: 30px; border: 1px solid rgba(255,255,255,0.1);
    text-align: center; font-family: 'Syne', sans-serif;
    max-width: 700px; margin: 30px auto;
}
.share-pct {
    font-size: 3em; font-weight: 800;
    background: linear-gradient(135deg, #ff6b6b, #ee5a24);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.share-line { color: #dfe6e9; margin: 8px 0; font-size: 0.9em; }
.share-tag  { color: #636e72; font-size: 0.85em; margin-top: 12px; }

.m-card {
    background: rgba(255,255,255,0.08); border-radius: 15px; padding: 20px; text-align: center;
    border: 1px solid rgba(255,255,255,0.1); margin: 6px auto; max-width: 250px;
}
.m-num { font-family: 'Syne', sans-serif; font-size: 1.8em; font-weight: 800; color: #fdcb6e; }
.m-lbl { font-size: 0.85em; color: #b2bec3; text-transform: uppercase; letter-spacing: 0.8px; }

.stProgress > div > div { border-radius: 99px; }
.block-container { padding-top: 3.5rem; max-width: 1300px; margin: auto; }

/* Mode buttons — uniform dark style */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: rgba(255,255,255,0.06) !important;
    color: #b2bec3 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
}

/* Hover effect for all buttons */
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: rgba(255,255,255,0.12) !important;
    border-color: rgba(255,255,255,0.28) !important;
    color: #fff !important;
    box-shadow: none !important;
}

/* Active mode button — dynamic color based on mode */
{''.join([f'div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"][data-baseweb="{key}"] {{\n    background: {color} !important;\n    border: 1px solid {color} !important;\n    color: #fff !important;\n    box-shadow: 0 0 12px {color} !important;\n}}\n' for key, color in zip(['normal', 'savage', 'emotional', 'delusional'], ['#c0392b', '#8e44ad', '#3498db', '#e67e22'])])}
</style>
""", unsafe_allow_html=True)

# ── Feature config ─────────────────────────────────────────────────────────────
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
    return ColumnTransformer([('num', num_t, NUM_FEATURES), ('cat', cat_t, CAT_FEATURES)], remainder='drop')

# ── Load / train models ────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rp = 'rf_reply_model.pkl'; gp = 'rf_ghost_model.pkl'; mp = 'model_metrics.pkl'
    if os.path.exists(rp) and os.path.exists(gp) and os.path.exists(mp):
        try:
            return joblib.load(rp), joblib.load(gp), joblib.load(mp)
        except Exception as e:
            st.warning(f"⚠️ Saved models failed ({str(e)[:60]}). Retraining...")

    with st.spinner("🤖 Training models... (~30 sec)"):
        try:
            df = pd.read_csv('ghosting_dataset5.csv')
        except FileNotFoundError:
            st.error("❌ ghosting_dataset5.csv not found. Run gen_data5.py first.")
            st.stop()

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

        def _train(df, target):
            X = df[NUM_FEATURES + CAT_FEATURES]; y = df[target]
            # ── Proper 3-way split ➺ no leakage ─────────────────────────────
            X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
            X_tr, X_val, y_tr, y_val   = train_test_split(X_tv, y_tv, test_size=0.15/0.85, random_state=42, stratify=y_tv)
            mdl = Pipeline([('pre', make_preprocessor()),
                            ('clf', RandomForestClassifier(n_estimators=400, max_depth=20,
                                class_weight='balanced', random_state=42, n_jobs=-1))])
            mdl.fit(X_tr, y_tr)
            # Calibrate on val only; evaluate on test only
            cal = CalibratedClassifierCV(mdl, method='sigmoid', cv=3)
            cal.fit(X_val, y_val)
            yp = cal.predict(X_test); yproba = cal.predict_proba(X_test)[:, 1]
            return cal, {
                'accuracy':  accuracy_score(y_test, yp),
                'precision': precision_score(y_test, yp, zero_division=0),
                'recall':    recall_score(y_test, yp, zero_division=0),
                'f1_score':  f1_score(y_test, yp, zero_division=0),
                'roc_auc':   roc_auc_score(y_test, yproba),
                'brier':     brier_score_loss(y_test, yproba),
                'confusion_matrix':      confusion_matrix(y_test, yp).tolist(),
                'classification_report': classification_report(y_test, yp),
                'train_size': len(X_tr), 'val_size': len(X_val), 'test_size': len(X_test),
                'y_test': y_test.tolist(), 'y_pred_prob': yproba.tolist(),
            }

        rm, rmets = _train(df, 'reply')
        gm, gmets = _train(df, 'ghosted')
        joblib.dump(rm, rp); joblib.dump(gm, gp)
        joblib.dump({'reply': rmets, 'ghosted': gmets}, mp)
        return rm, gm, {'reply': rmets, 'ghosted': gmets}

reply_model, ghost_model, all_metrics = load_models()

# ── Feature builder ────────────────────────────────────────────────────────────
def build_input(msg_len, tone, asked_q, resp_time, seen_ign, emoji,
                conv_len=25, rr=None, avg_rt=None, past_ghost=0, user_type='casual'):
    if rr is None:     rr     = 0.70 if asked_q else 0.50
    if avg_rt is None: avg_rt = max(1.0, resp_time * 0.5)
    tod = 'night' if resp_time > 20 else ('morning' if resp_time < 8 else 'day')
    return pd.DataFrame({
        'last_message_length':   [msg_len],
        'response_time_gap':     [float(resp_time)],
        'conversation_length':   [conv_len],
        'reply_ratio':           [rr],
        'avg_response_time':     [float(avg_rt)],
        'emoji_count':           [emoji],
        'question_asked':        [int(asked_q)],
        'seen_ignored':          [seen_ign],
        'past_ghosting_history': [past_ghost],
        'effort_score':          [msg_len + (emoji * 2) + (5 if asked_q else 0)],
        'delay':                 [0 if resp_time < 6 else (1 if resp_time < 24 else 2)],
        'is_dry':                [int(tone == 'dry')],
        'is_long_gap':           [int(resp_time > 24)],
        'engagement_score':      [rr * conv_len],
        'ghost_risk_combo':      [int(resp_time > 24 and rr < 0.4)],
        'seen_delay':            [int(seen_ign == 1 and resp_time > 12)],
        'initiator_flag':        [int(asked_q)],
        'inconsistency':         [int(abs(resp_time - avg_rt) > 20)],
        'decay_score':           [min(conv_len / 200, 1.0)],
        'effort_mismatch':       [int(msg_len > 20 and rr < 0.3)],
        'initiator':             ['me' if asked_q else 'them'],
        'message_tone':          [tone],
        'time_of_day':           [tod],
        'user_type':             [user_type],
    })

def predict_both(row):
    rp = reply_model.predict_proba(row)[0][1]
    gp = ghost_model.predict_proba(row)[0][1]
    return round(rp, 4), round(gp, 4)

# ── Mode-adjusted probabilities ───────────────────────────────────────────────
# The ML model gives one true probability. Each mode nudges it to tell a
# coherent story consistent with that mode's personality:
#   Savage:     slightly pessimistic ➺ surfaces the worst-case reading
#   Emotional:  softens ghost risk, because the point is empathy not alarm
#   Delusional: bumps reply up, tanks ghost risk ➺ the world is fine, always
#   Normal:     raw model output, no adjustment
#
# Adjustments are additive deltas, clamped to [0.05, 0.95].
# The base probability is always stored in session_state so switching modes
# always starts from the same model output ➺ no drift across mode switches.

MODE_PROB_DELTA = {
    #              reply_delta  ghost_delta
    'normal':    (  0.00,        0.00),
    'savage':    ( -0.07,       +0.10),   # pessimistic ➺ "realistically, it's worse"
    'emotional': ( +0.04,       -0.06),   # softer framing ➺ ghost risk feels lower
    'delusional':( +0.15,       -0.18),   # copium ➺ everything looks fine
}

def apply_mode(base_rp, base_gp, mode):
    rd, gd = MODE_PROB_DELTA[mode]
    rp = max(0.05, min(0.95, base_rp + rd))
    gp = max(0.05, min(0.95, base_gp + gd))
    return round(rp, 4), round(gp, 4)

# ── Mode-aware text ────────────────────────────────────────────────────────────
# BUG FIX: All text that changes with mode must be derived AFTER reading mode
# from session_state, and the cache key must include mode.

MODE_QUOTES = {
    'savage': {
        'high':  ("Not bad. They might actually respond. Don't ruin it by double-texting.",
                  "You're doing well. Shockingly."),
        'mid':   ("50/50. A coin toss. Even randomness has standards.",
                  "You're in the grey zone. Be honest ➺ you already know."),
        'low':   ("They saw it. Chose silence. That's your answer.",
                  "This isn't a delay. This is an exit."),
        'ghost_high': "They're already gone. The AI just confirmed what you felt.",
        'ghost_mid':  "It could go either way. But look at that response time.",
        'ghost_low':  "Slim chance. Still a chance. Do with that what you will.",
    },
    'emotional': {
        'high':  ("There's still warmth here. Don't give up on this connection 💛",
                  "The signs are good. You deserve someone who shows up."),
        'mid':   ("It's uncertain, and that uncertainty is exhausting. You're not alone in this.",
                  "You deserve clarity. This situation doesn't give you that yet."),
        'low':   ("It's okay to feel this. Silence hurts. Your feelings are valid.",
                  "Sometimes people fade. That's not a reflection of your worth."),
        'ghost_high': "This is hard to hear, but you already sensed something was off.",
        'ghost_mid':  "The uncertainty is real. You deserve better than wondering.",
        'ghost_low':  "There's still a thread here. But protect your heart either way.",
    },
    'delusional': {
        'high':  ("They're DEFINITELY writing a 3-paragraph reply right now 🔥",
                  "They literally can't stop thinking about you. Facts."),
        'mid':   ("They're just playing it cool. They're SO into you. Obviously.",
                  "This is called mystery. They're keeping you guessing because you're special."),
        'low':   ("They're probably just in a coma. Or lost their phone. In the ocean.",
                  "WiFi issues. 100%. They'll reply any second now. Any. Second."),
        'ghost_high': "Ghost risk?? No no no. They're just... composing the perfect reply.",
        'ghost_mid':  "The model is clearly broken. You two have something special.",
        'ghost_low':  "See?? Low ghost risk. They adore you. Manifesting the reply rn.",
    },
    'normal': {
        'high':  ("Good signs based on your inputs. Message has solid energy.",
                  "The indicators are positive here."),
        'mid':   ("This one could genuinely go either way. Hard to call.",
                  "Mixed signals in the data ➺ reply is uncertain."),
        'low':   ("The probability here is low based on current signals.",
                  "Several risk factors are stacking up in this scenario."),
        'ghost_high': "Multiple ghosting indicators are present.",
        'ghost_mid':  "Some ghosting signals detected ➺ not conclusive.",
        'ghost_low':  "Low ghosting probability based on the inputs.",
    }
}

# BUG FIX: Final Verdict text must also be mode-aware
VERDICT_TEXT = {
    'normal': {
        'clear_ok':    ("You're overcomplicating this. They'll reply.", "#00b894"),
        'mixed':       ("Mixed signals. Reply likely but something feels off.", "#fdcb6e"),
        'one_sided':   ("This is one-sided. You're investing more than they are.", "#e17055"),
        'move_on':     ("Move on. The data agrees with your gut.", "#d63031"),
        'uncertain':   ("It's uncertain. Give it one more day before deciding.", "#636e72"),
    },
    'savage': {
        'clear_ok':    ("They'll reply. Don't sabotage it now.", "#00b894"),
        'mixed':       ("Reply likely. Ghost possible. Classic mixed energy situation.", "#fdcb6e"),
        'one_sided':   ("You're the only one putting in effort here. Read that again.", "#e17055"),
        'move_on':     ("It's over. Your gut knew. Now you have data too.", "#d63031"),
        'uncertain':   ("Genuinely unclear. But your anxiety already picked a side.", "#636e72"),
    },
    'emotional': {
        'clear_ok':    ("There's real connection here. Let it breathe.", "#00b894"),
        'mixed':       ("Something good is here, but something's also holding back.", "#fdcb6e"),
        'one_sided':   ("You deserve reciprocity. This doesn't look balanced right now.", "#e17055"),
        'move_on':     ("It's okay to let go. That's not giving up, it's self-respect.", "#d63031"),
        'uncertain':   ("Uncertainty is painful. Whatever happens, you'll be okay.", "#636e72"),
    },
    'delusional': {
        'clear_ok':    ("Obviously they'll reply. You two are basically soulmates.", "#00b894"),
        'mixed':       ("The universe is just building tension before the plot twist 🌟", "#fdcb6e"),
        'one_sided':   ("You're the main character. They're just processing their feelings.", "#e17055"),
        'move_on':     ("'Move on'?? The AI doesn't understand your unique connection.", "#d63031"),
        'uncertain':   ("The model is just shy. It doesn't understand romance.", "#636e72"),
    },
}

def get_verdict_key(reply_prob, ghost_prob):
    if reply_prob > 0.65 and ghost_prob < 0.40:  return 'clear_ok'
    if reply_prob > 0.65 and ghost_prob >= 0.40: return 'mixed'
    if reply_prob > 0.40:                         return 'one_sided'
    if ghost_prob > 0.65:                         return 'move_on'
    return 'uncertain'

def get_quotes(mode, reply_prob, ghost_prob):
    rb = 'high' if reply_prob > 0.65 else ('mid' if reply_prob > 0.40 else 'low')
    gb = 'ghost_high' if ghost_prob > 0.65 else ('ghost_mid' if ghost_prob > 0.40 else 'ghost_low')
    q = MODE_QUOTES[mode][rb]
    return (q[0], q[1]), MODE_QUOTES[mode][gb]

def interest_label(reply_prob):
    if reply_prob > 0.70: return "HIGH", "val-high"
    if reply_prob > 0.45: return "MEDIUM", "val-mid"
    return "LOW", "val-low"

def effort_label(msg_len, asked_q, emoji):
    score = (msg_len / 50) + (2 if asked_q else 0) + (emoji * 0.3)
    if score > 4: return "HIGH", "val-high"
    if score > 2: return "BALANCED", "val-mid"
    return "ONE-SIDED", "val-low"

def ghost_risk_label(ghost_prob):
    if ghost_prob > 0.65: return "HIGH", "val-low"
    if ghost_prob > 0.40: return "MEDIUM", "val-mid"
    return "LOW", "val-high"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>💀 GHOSTING PREDICTOR</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered relationship reality check ➺ be honest, it already knows</div>", unsafe_allow_html=True)

# ── Personality mode selector ──────────────────────────────────────────────────
st.markdown("#### Choose your vibe")
if "personality_mode" not in st.session_state:
    st.session_state["personality_mode"] = "normal"

mode_cols = st.columns(4)
modes = [("🧠 Normal", "normal"), ("💀 Savage", "savage"), ("😭 Emotional", "emotional"), ("🤡 Delusional", "delusional")]
for i, (lbl, key) in enumerate(modes):
    with mode_cols[i]:
        if st.button(lbl, use_container_width=True,
                     type="primary" if st.session_state["personality_mode"] == key else "secondary"):
            if st.session_state.get("last_clicked") == key:
                # Double-click detected, toggle to red
                st.session_state["personality_mode"] = "savage"
            else:
                st.session_state["personality_mode"] = key
            st.session_state["last_clicked"] = key
            # NO st.rerun() here. Streamlit re-renders naturally on button click.
            # st.rerun() caused a second render cycle where tab widgets hadn't
            # initialized yet — ikey matched stale session values and base_rp/base_gp
            # fell back to mode defaults instead of the user's actual inputs.

# Read mode ONCE here ➺ everything below uses this single variable
mode = st.session_state["personality_mode"]

mode_banner = {
    "normal":     ("🧠 Normal Mode", "#636e72"),
    "savage":     ("💀 Savage Mode ➺ No feelings were harmed. They were obliterated.", "#d63031"),
    "emotional":  ("😭 Emotional Mode ➺ We see you. Your feelings are valid.", "#6c5ce7"),
    "delusional": ("🤡 Delusional Mode ➺ Stay hopeful! (AI thinks you're cooked.)", "#e17055"),
}
banner_text, banner_color = mode_banner[mode]
st.markdown(
    f"<div style='text-align:center;background:{banner_color}22;border:1px solid {banner_color}55;"
    f"border-radius:10px;padding:8px;font-size:1.5em;color:{banner_color};margin:8px 0 16px;'>"
    f"{banner_text}</div>",
    unsafe_allow_html=True
)
st.divider()

# ── Tabs: 3 tabs only (message analyzer removed) ──────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 What-If", "🎓 Model Metrics"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 ➺ MAIN PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 📱 Your message")
        message_length = st.slider("Message length (chars)", 1, 500, 80, 5)
        message_tone   = st.selectbox("Tone", ['dry', 'neutral', 'enthusiastic'], index=1)
        asked_question = st.toggle("Asked a question?", value=True)
        emoji_count    = st.slider("Emojis used", 0, 10, 1)
        past_ghost     = st.toggle("Have they ghosted you before?", value=False)

    with col2:
        st.markdown("#### ⏱️ Their behaviour")
        response_time  = st.slider("Hours since you sent it", 0, 72, 4, 1)
        seen_raw       = st.radio("Did they see it?", ["👁️ Yes, seen", "❓ Not seen yet"], horizontal=True)
        seen_ignored   = 1 if "Yes" in seen_raw else 0
        conv_len       = st.slider("How long has the convo been? (messages)", 1, 200, 20)
        user_type_map  = {
            "😊 Seems interested": "interested",
            "💬 Normal/casual":    "casual",
            "🌵 Very dry texter":  "dry_texter",
            "👻 Known to ghost":   "ghoster",
        }
        user_type_label = st.selectbox("How would you describe them?", list(user_type_map.keys()))
        user_type       = user_type_map[user_type_label]

    st.divider()

    # ── Predictions ➺ cache key includes mode so text refreshes on mode change ──
    # ikey tracks input changes only (not mode) ➺ model is only re-run when
    # inputs change. Base probabilities are stored raw (no mode applied).
    # Mode adjustment is applied on every render so switching mode instantly
    # changes the displayed numbers without re-running the model.
    ikey = (message_length, message_tone, asked_question, response_time,
            seen_ignored, emoji_count, conv_len, user_type, int(past_ghost))

    # Always run prediction if base_rp/base_gp are missing OR if inputs changed
    needs_predict = (
        st.session_state.get("ikey") != ikey
        or "base_rp" not in st.session_state
        or "base_gp" not in st.session_state
    )

    if needs_predict:
        try:
            row = build_input(message_length, message_tone, asked_question,
                              response_time, seen_ignored, emoji_count,
                              conv_len=conv_len, past_ghost=int(past_ghost),
                              user_type=user_type)
            base_rp, base_gp = predict_both(row)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            base_rp, base_gp = 0.5, 0.4
        st.session_state.update({"ikey": ikey, "base_rp": base_rp, "base_gp": base_gp})

    # Apply mode delta on every render ➺ no model re-run needed
    reply_prob, ghost_prob = apply_mode(
        st.session_state["base_rp"],
        st.session_state["base_gp"],
        mode
    )

    # Derive ALL mode-dependent text here, after reading mode from session_state
    (main_q, sub_q), ghost_q = get_quotes(mode, reply_prob, ghost_prob)
    verdict_key = get_verdict_key(reply_prob, ghost_prob)
    verdict_text, verdict_color = VERDICT_TEXT[mode][verdict_key]

    # ── Dual verdict cards ────────────────────────────────────────────────────
    vc1, vc2 = st.columns(2)
    with vc1:
        vclass = "verdict-high" if reply_prob > 0.65 else ("verdict-mid" if reply_prob > 0.40 else "verdict-low")
        vlabel = "They'll reply 🔥" if reply_prob > 0.65 else ("Could go either way 😬" if reply_prob > 0.40 else "They're ghosting you 💀")
        st.markdown(f"""
        <div class='verdict-card {vclass}'>
            <div style='font-size:0.8em;font-weight:600;opacity:0.8;text-transform:uppercase;letter-spacing:1px;'>Reply probability</div>
            <div class='verdict-pct'>{reply_prob*100:.0f}%</div>
            <div class='verdict-label'>{vlabel}</div>
            <div class='verdict-quote'>"{main_q}"</div>
        </div>
        """, unsafe_allow_html=True)

    with vc2:
        gclass = "verdict-low" if ghost_prob > 0.65 else ("verdict-mid" if ghost_prob > 0.40 else "verdict-high")
        glabel = "High ghost risk 💀" if ghost_prob > 0.65 else ("Uncertain 😬" if ghost_prob > 0.40 else "Probably fine 🙂")
        st.markdown(f"""
        <div class='verdict-card {gclass}'>
            <div style='font-size:0.8em;font-weight:600;opacity:0.8;text-transform:uppercase;letter-spacing:1px;'>Ghost probability</div>
            <div class='verdict-pct'>{ghost_prob*100:.0f}%</div>
            <div class='verdict-label'>{glabel}</div>
            <div class='verdict-quote'>"{ghost_q}"</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Conversation Diagnosis ────────────────────────────────────────────────
    # BUG FIX: Diagnosis values are derived from ML probabilities (correct),
    # but the Read Status now also reflects mode tone
    st.markdown("#### 🧠 Conversation Diagnosis")
    int_lbl, int_cls = interest_label(reply_prob)
    eff_lbl, eff_cls = effort_label(message_length, asked_question, emoji_count)
    gr_lbl,  gr_cls  = ghost_risk_label(ghost_prob)

    # Read status changes with mode (delusional gives an excuse, savage is blunt)
    if seen_ignored and response_time > 6:
        seen_txt = {
            'normal':    "IGNORED 🚨",
            'savage':    "SEEN. IGNORED. 💀",
            'emotional': "SEEN, NO REPLY 💔",
            'delusional':"SEEN (composing!!) ✍️",
        }[mode]
        seen_cls = "val-low"
    elif seen_ignored:
        seen_txt = "SEEN ✓"; seen_cls = "val-mid"
    else:
        seen_txt = "NOT SEEN"; seen_cls = "val-mid"

    st.markdown(f"""
    <div class='diag-row'>
        <div class='diag-card'>
            <div class='diag-title'>Interest Level</div>
            <div class='diag-val {int_cls}'>{int_lbl}</div>
        </div>
        <div class='diag-card'>
            <div class='diag-title'>Effort Balance</div>
            <div class='diag-val {eff_cls}'>{eff_lbl}</div>
        </div>
        <div class='diag-card'>
            <div class='diag-title'>Ghost Risk</div>
            <div class='diag-val {gr_cls}'>{gr_lbl}</div>
        </div>
        <div class='diag-card'>
            <div class='diag-title'>Read Status</div>
            <div class='diag-val {seen_cls}'>{seen_txt}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Signal Breakdown ──────────────────────────────────────────────────────
    st.markdown("#### 🚩 Signal Breakdown")
    fc1, fc2 = st.columns(2)

    red_flags, green_flags = [], []
    if seen_ignored and response_time > 6:   red_flags.append("They saw your message. They chose silence.")
    if response_time > 48:                   red_flags.append(f"It's been {response_time}h. That's not busy, that's avoidance.")
    elif response_time > 24:                 red_flags.append("Over 24 hours ➺ the energy is cooling off.")
    if message_tone == 'dry':                red_flags.append("Dry tone doesn't open doors.")
    if not asked_question:                   red_flags.append("No question = no reason to reply.")
    if message_length < 30:                  red_flags.append("Short message ➺ looks like low effort.")
    if user_type == 'ghoster':               red_flags.append("You described them as a known ghoster. That's data.")
    if past_ghost:                           red_flags.append("They've ghosted you before. Pattern recognised.")
    if emoji_count == 0 and message_tone == 'dry': red_flags.append("Zero warmth signals in this message.")

    if asked_question:                       green_flags.append("Asked a question ➺ gives them something to respond to.")
    if message_length > 100:                 green_flags.append("Substantial message ➺ shows you put in effort.")
    if message_tone == 'enthusiastic':       green_flags.append("Enthusiastic tone ➺ energy is contagious.")
    if response_time < 12:                   green_flags.append("Sent recently ➺ they still might be composing a reply.")
    if emoji_count > 0:                      green_flags.append("Used emojis ➺ lightens the vibe.")
    if user_type == 'interested':            green_flags.append("You described them as interested ➺ that matters.")
    if not past_ghost:                       green_flags.append("No ghosting history ➺ fresh start.")

    with fc1:
        st.markdown("**Red flags**")
        for f in red_flags:
            st.markdown(f"<div class='flag-item'>🚩 {f}</div>", unsafe_allow_html=True)
        if not red_flags:
            st.markdown("<div class='flag-item green-flag'>✅ No major red flags detected.</div>", unsafe_allow_html=True)

    with fc2:
        st.markdown("**Green flags**")
        for g in green_flags:
            st.markdown(f"<div class='flag-item green-flag'>✅ {g}</div>", unsafe_allow_html=True)
        if not green_flags:
            st.markdown("<div class='flag-item'>🚩 Hmm, not many positives here.</div>", unsafe_allow_html=True)

    st.divider()

    # ── Final Verdict ➺ mode-aware ────────────────────────────────────────────
    # BUG FIX: verdict_text and verdict_color now come from VERDICT_TEXT[mode]
    st.markdown("#### 🎯 Final Verdict")
    st.markdown(
        f"<div style='background:{verdict_color}22;border-left:4px solid {verdict_color};"
        f"border-radius:0 12px 12px 0;padding:16px 20px;margin:10px 0;"
        f"font-family:Syne,sans-serif;font-size:1.1em;color:{verdict_color};font-weight:600;'>"
        f"{verdict_text}</div>",
        unsafe_allow_html=True
    )
    if sub_q:
        st.markdown(
            f"<div style='color:#b2bec3;font-style:italic;font-size:0.9em;margin-top:8px;'>💭 {sub_q}</div>",
            unsafe_allow_html=True
        )

    st.divider()

    # ── Shareable card ────────────────────────────────────────────────────────
    st.markdown("#### 📸 Share Your Result")
    share_label = "They'll probably reply 🔥" if reply_prob > 0.65 else ("It's a coin flip 😬" if reply_prob > 0.40 else "Ghosting incoming 💀")
    ghost_label = "Ghost risk: HIGH 💀" if ghost_prob > 0.65 else ("Ghost risk: MEDIUM ⚠️" if ghost_prob > 0.40 else "Ghost risk: LOW ✅")

    st.markdown(f"""
    <div class='share-card'>
        <div style='font-size:1.75em;letter-spacing:2px;color:#636e72;text-transform:uppercase;margin-bottom:8px;'>AI Reality Check</div>
        <div class='share-pct'>{reply_prob*100:.0f}%</div>
        <div class='share-line' style='font-size:1.2em;font-weight:700;'>{share_label}</div>
        <div class='share-line' style='color:#b2bec3;'>{ghost_label}</div>
        <div class='share-line' style='font-style:italic;color:#dfe6e9;margin-top:10px;font-size:1.95em;'>"{main_q}"</div>
        <div class='share-tag'>#GhostingPredictor • ghostingpredictor.app</div>
    </div>
    """, unsafe_allow_html=True)

    share_text = (f"💀 Ghosting Predictor says:\n"
                  f"Reply chance: {reply_prob*100:.0f}% ➺ {share_label}\n"
                  f"{ghost_label}\n"
                  f'"{main_q}"\n'
                  f"#GhostingPredictor #AI #Dating")

    # FIX: st.button causes a full page rerun which resets widget defaults →
    # changes ikey → triggers fresh prediction with default inputs → wrong %.
    # Solution: always render the share text in a st.text_area (read-only style).
    # st.text_area does NOT trigger a rerun when the user clicks inside it to
    # select/copy ➺ it only reruns on actual value change, which can't happen
    # because the value is set programmatically and the user just selects text.
    st.markdown("<div style='font-size:0.82em;color:#b2bec3;margin-bottom:4px;'>📋 Click inside, Ctrl+A, Ctrl+C to copy:</div>", unsafe_allow_html=True)
    st.text_area(
        label="share_text_area",
        value=share_text,
        height=120,
        label_visibility="collapsed",
        key=f"share_ta_{hash(share_text)}",   # stable key tied to content, not mode
    )
    st.markdown("<div style='text-align:center;font-size:1.8em;color:#636e72;margin-top:4px;'>📸 Or screenshot the card above and post it</div>", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;margin-top:12px;font-size:0.85em;color:#636e72;'>Drop your situation in the comments ➺ I'll tell you what the model says 👇</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 ➺ WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 📊 What-If Simulator")
    st.markdown("<div style='color:#636e72;font-size:1.5em;'>See how your odds change if you tweak one thing. Experiment freely.</div>", unsafe_allow_html=True)

    if "base_rp" not in st.session_state:
        st.info("Go to the **Predict** tab first to set your base scenario.")
    else:
        base_rp, base_gp = apply_mode(
            st.session_state["base_rp"],
            st.session_state["base_gp"],
            mode
        )
        ml, tone, aq, rt, si, ec, cl_val, ut, pg = st.session_state["ikey"]

        rclr = "#00b894" if base_rp > 0.65 else ("#fdcb6e" if base_rp > 0.4 else "#ff7675")
        gclr = "#ff7675" if base_gp > 0.65 else ("#fdcb6e" if base_gp > 0.4 else "#00b894")
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.05);border-radius:12px;padding:14px 18px;margin-bottom:16px;'>
            <div style='font-size:0.8em;color:#b2bec3;text-transform:uppercase;letter-spacing:1px;'>Your current situation</div>
            <div style='font-family:Syne,sans-serif;font-size:1.6em;font-weight:700;'>
                Reply: <span style='color:{rclr}'>{base_rp*100:.0f}%</span>
                &nbsp;&nbsp; Ghost: <span style='color:{gclr}'>{base_gp*100:.0f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        scenarios = []
        if not aq:
            r = build_input(ml, tone, True, rt, si, ec, cl_val, user_type=ut)
            nr, ng = predict_both(r)
            scenarios.append(("❓ If you added a question", nr, ng))
        if tone != 'enthusiastic':
            r = build_input(ml, 'enthusiastic', aq, rt, si, ec, cl_val, user_type=ut)
            nr, ng = predict_both(r)
            scenarios.append(("😄 If your tone was enthusiastic", nr, ng))
        if ml < 150:
            r = build_input(150, tone, aq, rt, si, ec, cl_val, user_type=ut)
            nr, ng = predict_both(r)
            scenarios.append(("📝 If your message was longer (150 chars)", nr, ng))
        if rt > 12:
            r = build_input(ml, tone, aq, 2, si, ec, cl_val, user_type=ut)
            nr, ng = predict_both(r)
            scenarios.append(("⏱️ If you followed up now (2h gap)", nr, ng))
        if ec == 0:
            r = build_input(ml, tone, aq, rt, si, 3, cl_val, user_type=ut)
            nr, ng = predict_both(r)
            scenarios.append(("😂 If you added 3 emojis", nr, ng))
        r = build_input(max(ml, 120), 'enthusiastic', True, min(rt, 4), si, max(ec, 2), cl_val, user_type=ut)
        nr, ng = predict_both(r)
        scenarios.append(("🚀 Best case (all fixes applied)", nr, ng))

        st.markdown("**How your odds change:**")
        for label, nr, ng in scenarios:
            rdiff = (nr - base_rp) * 100
            gdiff = (ng - base_gp) * 100
            rc = "whatif-boost" if rdiff > 0 else "whatif-drop"
            gc = "whatif-drop"  if gdiff > 0 else "whatif-boost"
            rs = "+" if rdiff >= 0 else ""; gs = "+" if gdiff >= 0 else ""
            st.markdown(f"""
            <div class='whatif-row'>
                <span>{label}</span>
                <span>
                    <span class='{rc}'>Reply: {rs}{rdiff:.0f}%</span>
                    &nbsp;|&nbsp;
                    <span class='{gc}'>Ghost: {gs}{gdiff:.0f}%</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='color:#636e72;font-size:1.5em;margin-top:12px;'>All scenarios keep the rest of your inputs unchanged.</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 ➺ MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 🎓 Model Performance")
    st.markdown("<div style='color:#636e72;font-size:1.5em;'>Two calibrated Random Forest models ➺ reply prediction and ghost prediction. Evaluated on a clean held-out test set.</div>", unsafe_allow_html=True)

    for key, label in [('reply', '📩 Reply Model'), ('ghosted', '👻 Ghost Model')]:
        m = all_metrics[key]
        st.markdown(f"### {label}")
        mc = st.columns(5)
        mc[0].markdown(f"<div class='m-card'><div class='m-num'>{m['accuracy']*100:.1f}%</div><div class='m-lbl'>Accuracy</div></div>", unsafe_allow_html=True)
        mc[1].markdown(f"<div class='m-card'><div class='m-num'>{m['precision']*100:.1f}%</div><div class='m-lbl'>Precision</div></div>", unsafe_allow_html=True)
        mc[2].markdown(f"<div class='m-card'><div class='m-num'>{m['recall']*100:.1f}%</div><div class='m-lbl'>Recall</div></div>", unsafe_allow_html=True)
        mc[3].markdown(f"<div class='m-card'><div class='m-num'>{m['f1_score']:.3f}</div><div class='m-lbl'>F1 Score</div></div>", unsafe_allow_html=True)
        mc[4].markdown(f"<div class='m-card'><div class='m-num'>{m['roc_auc']:.3f}</div><div class='m-lbl'>ROC-AUC</div></div>", unsafe_allow_html=True)

        with st.expander(f"Confusion matrix & report ➺ {label}", expanded=False):
            e1, e2 = st.columns(2)
            with e1:
                cm_arr = np.array(m['confusion_matrix'])
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                            ax=ax, cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
                ax.set_xlabel('Predicted', color='white'); ax.set_ylabel('Actual', color='white')
                ax.set_title('Confusion Matrix', color='white', fontsize=11)
                fig.patch.set_facecolor('#1a1a2e'); ax.set_facecolor('#1a1a2e')
                ax.tick_params(colors='white')
                plt.tight_layout(); st.pyplot(fig, use_container_width=True)
            with e2:
                try:
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(np.array(m['y_test']), np.array(m['y_pred_prob']))
                    fig2, ax2 = plt.subplots(figsize=(4, 3))
                    ax2.plot(fpr, tpr, color='#ee5a24', lw=2, label=f"AUC={m['roc_auc']:.3f}")
                    ax2.plot([0,1],[0,1],'--',color='gray',lw=1)
                    ax2.fill_between(fpr, tpr, alpha=0.12, color='#ee5a24')
                    ax2.set_xlabel('FPR', color='white'); ax2.set_ylabel('TPR', color='white')
                    ax2.set_title('ROC Curve', color='white', fontsize=11)
                    ax2.legend(fontsize=9); ax2.tick_params(colors='white')
                    ax2.set_facecolor('#1a1a2e'); fig2.patch.set_facecolor('#1a1a2e')
                    plt.tight_layout(); st.pyplot(fig2, use_container_width=True)
                except: pass
            st.code(m['classification_report'], language=None)
        st.divider()

    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.04);border-radius:12px;padding:16px 20px;font-size:1.0em;color:#636e72;'>
    <b>Architecture:</b> Random Forest (400 trees, depth=20, class_weight=balanced) + Sigmoid calibration (cv=3)<br>
    <b>Split:</b> 70% train / 15% calibration val / 15% test ➺ no data leakage between steps<br>
    <b>Dataset:</b> 10,000 synthetic samples ➺ ghosting_dataset5.csv<br>
    <b>Features:</b> 20 numerical + 4 categorical (including user_type persona)
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#636e72;margin-top:40px;padding:20px;font-size:1.85em;'>
    <div style='margin-bottom:4px;'>💭 <i>You already know the answer. The AI just confirmed it.</i></div>
    <div>Powered by Random Forest ML · Not liable for heartbreak 💔</div>
</div>
""", unsafe_allow_html=True)