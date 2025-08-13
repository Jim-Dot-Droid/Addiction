# Fair Crash Game Statistical Predictor

This Streamlit app predicts the most likely time to bet in a 100% fair crash game (p=0.5 for under 2.0x).
It uses only historical under counts vs expected to detect mean reversion.

## How it works
- Looks at last N rounds (default: 20)
- Counts unders (<2.0x)
- Compares to expected under count in a fair game
- Calculates Z-score and probability of over next
- Predicts 'Above' if probability > 55%, else 'Under'

## Run
```
pip install -r requirements.txt
streamlit run app.py
```