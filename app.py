
import streamlit as st
import pandas as pd
import numpy as np
import os
from math import sqrt
from scipy.stats import norm

# File paths
HISTORY_FILE = "history.csv"
RESULTS_FILE = "results.csv"
BALANCE_FILE = "sol_balance.txt"

# Config
INITIAL_BALANCE = 0.1
BET_AMOUNT = 0.02
WINDOW = 20  # last N rounds
FAIR_UNDER_RATE = 0.5  # 100% fair crash game

@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    if 'multiplier' in df.columns:
        return df['multiplier'].tolist()
    return df.iloc[:, 0].tolist()

def load_history():
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        return df['multiplier'].tolist()
    return []

def save_history(data):
    pd.DataFrame({'multiplier': data}).to_csv(HISTORY_FILE, index=False)

def load_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame(columns=['prediction', 'actual', 'correct'])

def save_result(prediction, actual):
    correct = ((prediction == "Above") and actual > 2.0) or ((prediction == "Under") and actual <= 2.0)
    results_df = load_results()
    results_df.loc[len(results_df)] = [prediction, actual, correct]
    results_df.to_csv(RESULTS_FILE, index=False)
    update_balance(prediction, actual)

def get_balance():
    if os.path.exists(BALANCE_FILE):
        return float(open(BALANCE_FILE).read())
    return INITIAL_BALANCE

def update_balance(prediction, actual):
    balance = get_balance()
    if prediction == "Above":
        balance += BET_AMOUNT if actual > 2.0 else -BET_AMOUNT
    with open(BALANCE_FILE, "w") as f:
        f.write(str(balance))

def reset_all():
    for f in [HISTORY_FILE, RESULTS_FILE, BALANCE_FILE]:
        if os.path.exists(f):
            os.remove(f)

def normalize_input(value):
    return value / 100 if value > 10 else value

def fair_game_prediction(data, window=WINDOW, threshold=2.0, fair_under_rate=FAIR_UNDER_RATE):
    if len(data) < window:
        return None

    recent = np.array(data[-window:])
    under_count = np.sum(recent < threshold)
    expected_mean = window * fair_under_rate
    std_dev = sqrt(window * fair_under_rate * (1 - fair_under_rate))

    z_score = (under_count - expected_mean) / std_dev if std_dev > 0 else 0.0
    prob_over_next = 1 - norm.cdf(z_score)

    return {
        "under_count": int(under_count),
        "expected_mean": round(expected_mean, 2),
        "std_dev": round(std_dev, 3),
        "z_score": round(z_score, 3),
        "prob_over_next": round(prob_over_next * 100, 2),
        "prediction": "Above" if prob_over_next > 0.55 else "Under"
    }

def main():
    st.title("🎯 Fair Crash Game Statistical Predictor")
    st.caption("Predicts based purely on deviation from expected under rate in a 100% fair crash game (p=0.5).")

    if "history" not in st.session_state:
        st.session_state.history = load_history()

    # Upload CSV
    uploaded = st.file_uploader("Upload multipliers CSV", type=["csv"])
    if uploaded:
        st.session_state.history = load_csv(uploaded)
        save_history(st.session_state.history)
        st.success(f"Loaded {len(st.session_state.history)} multipliers.")

    # Manual input
    new_val = st.text_input("Enter new multiplier (e.g., 1.87 or 187)")
    if st.button("Add multiplier"):
        try:
            val = normalize_input(float(new_val))
            if "last_prediction" in st.session_state:
                save_result(st.session_state.last_prediction, val)
                del st.session_state.last_prediction
            st.session_state.history.append(val)
            save_history(st.session_state.history)
            st.success(f"Added {val}x")
        except:
            st.error("Invalid number.")

    # Reset
    if st.button("Reset All Data"):
        st.session_state.history = []
        reset_all()
        st.success("All data reset.")

    # Prediction
    if st.session_state.history:
        st.write(f"History length: **{len(st.session_state.history)}**")
        pred = fair_game_prediction(st.session_state.history)
        if pred:
            st.subheader("📊 Fair Game Stats")
            st.write(f"Unders in last {WINDOW}: **{pred['under_count']}** (expected {pred['expected_mean']})")
            st.write(f"Std Dev: **{pred['std_dev']}**")
            st.write(f"Z-Score: **{pred['z_score']}**")
            st.write(f"Prob Over Next: **{pred['prob_over_next']}%**")
            st.session_state.last_prediction = pred['prediction']
            if pred['prediction'] == "Above":
                st.success("Prediction: **Above 2.0x** ✅")
            else:
                st.warning("Prediction: **Under 2.0x** ❌")
        else:
            st.info(f"Need at least {WINDOW} data points to predict.")
    else:
        st.write("No data yet.")

    # Results tracker
    st.subheader("📈 Accuracy Tracker")
    results_df = load_results()
    if not results_df.empty:
        total = len(results_df)
        correct = int(results_df['correct'].sum())
        acc = correct / total if total else 0
        st.metric("Total Predictions", total)
        st.metric("Correct Predictions", correct)
        st.metric("Accuracy Rate", f"{acc:.1%}")
        st.dataframe(results_df[::-1].reset_index(drop=True))
    else:
        st.write("No predictions verified yet.")

    # Balance
    st.subheader("💰 SOL Balance Tracker")
    st.metric("Balance", f"{get_balance():.4f} SOL")
    st.caption(f"Flat betting {BET_AMOUNT} SOL when 'Above'. Starting balance: {INITIAL_BALANCE} SOL.")

if __name__ == "__main__":
    main()
