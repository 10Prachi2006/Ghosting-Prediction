import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

n = 10000

data = []

for _ in range(n):

    # --- RAW FEATURES ---
    last_message_length  = random.randint(1, 50)
    response_time_gap    = random.uniform(0, 72)
    initiator            = random.choice(['me', 'them'])
    conversation_length  = random.randint(1, 200)
    reply_ratio          = round(random.uniform(0, 1), 2)
    avg_response_time    = random.uniform(1, 120)
    message_tone         = random.choice(['dry', 'neutral', 'enthusiastic'])
    emoji_count          = random.randint(0, 10)
    question_asked       = random.choice([0, 1])
    time_of_day          = random.choice(['day', 'night'])
    seen_ignored         = random.choice([0, 1])
    past_ghosting_history = random.choice([0, 1])

    # --- PERSONA (KEY IDEA FROM CHATGPT — BUT EXPOSE IT AS A FEATURE) ---
    user_type = random.choice(['interested', 'casual', 'dry_texter', 'ghoster'])

    # --- REPLY LOGIC ---
    reply_prob = 0.5
    if seen_ignored == 1 and response_time_gap > 24: reply_prob -= 0.35
    if message_tone == 'enthusiastic':               reply_prob += 0.20
    if question_asked == 1:                          reply_prob += 0.15
    if response_time_gap > 24:                       reply_prob -= 0.20

    # Persona effect on reply
    if user_type == 'interested':  reply_prob += 0.25
    elif user_type == 'casual':    reply_prob += 0.05
    elif user_type == 'dry_texter': reply_prob -= 0.10
    elif user_type == 'ghoster':   reply_prob -= 0.30

    reply_prob = max(0.05, min(reply_prob, 0.95))
    reply = 1 if random.random() < reply_prob else 0

    # --- GHOSTING LOGIC ---
    ghost_prob = 0.20

    # Response time (capped contribution)
    if response_time_gap > 48:   ghost_prob += 0.28
    elif response_time_gap > 24: ghost_prob += 0.15

    # Reply ratio (capped contribution)
    if reply_ratio < 0.2:        ghost_prob += 0.22
    elif reply_ratio < 0.4:      ghost_prob += 0.12

    # Tone
    if message_tone == 'dry':     ghost_prob += 0.15
    elif message_tone == 'neutral': ghost_prob += 0.05

    # Seen ignored
    if seen_ignored == 1:         ghost_prob += 0.18

    # Past ghosting
    if past_ghosting_history == 1: ghost_prob += 0.15

    # Combo: high gap + low engagement (ChatGPT step 3 — done safely)
    # Cap this so it can't stack to 0.98 by itself
    combo_bonus = 0.0
    if reply_ratio < 0.3 and response_time_gap > 24:
        combo_bonus += 0.15
    # Conversation decay (ChatGPT step 3 — made safe)
    # Only applies a small, capped bonus — not a runaway multiplier
    decay = min(conversation_length / 200, 1.0)   # max 1.0
    combo_bonus += decay * 0.10                     # max +0.10, not +0.30

    # Emotional mismatch (ChatGPT step 4)
    if message_tone == 'enthusiastic' and reply_ratio < 0.3:
        combo_bonus += 0.08

    # Inconsistency (ChatGPT step 3)
    if abs(response_time_gap - avg_response_time) > 20:
        combo_bonus += 0.08

    # HARD CAP on combo to prevent stacking explosion
    combo_bonus = min(combo_bonus, 0.25)
    ghost_prob += combo_bonus

    # Persona effect on ghosting
    if user_type == 'interested':   ghost_prob -= 0.18
    elif user_type == 'casual':     ghost_prob += 0.05
    elif user_type == 'dry_texter': ghost_prob += 0.12
    elif user_type == 'ghoster':    ghost_prob += 0.28

    ghost_prob = max(0.05, min(ghost_prob, 0.92))   # cap at 0.92, not 0.98
    ghosted = 1 if random.random() < ghost_prob else 0

    # 2% label noise
    if random.random() < 0.02: reply   = 1 - reply
    if random.random() < 0.02: ghosted = 1 - ghosted

    data.append([
        last_message_length, response_time_gap, initiator,
        conversation_length, reply_ratio, avg_response_time,
        message_tone, emoji_count, question_asked, time_of_day,
        seen_ignored, past_ghosting_history, user_type,  # <-- exposed!
        reply, ghosted
    ])

columns = [
    'last_message_length', 'response_time_gap', 'initiator',
    'conversation_length', 'reply_ratio', 'avg_response_time',
    'message_tone', 'emoji_count', 'question_asked', 'time_of_day',
    'seen_ignored', 'past_ghosting_history',
    'user_type',  # persona is now a feature the model can learn from
    'reply', 'ghosted'
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("ghosting_dataset5.csv", index=False)

print(df.shape)
print("\nClass distribution (ghosted):")
print(df['ghosted'].value_counts())
print(f"Ghosting rate: {df['ghosted'].mean():.1%}")
print("\nClass distribution (reply):")
print(df['reply'].value_counts())
print(f"Reply rate: {df['reply'].mean():.1%}")
print("\nUser type distribution:")
print(df['user_type'].value_counts())