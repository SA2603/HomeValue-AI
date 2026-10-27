# HomeValue.ai: House Price Prediction System

A machine-learning web app that estimates a home's market value, with a realistic price range.

## Features
- Gradient Boosting model trained on the California Housing dataset
- Feature engineering (bedroom ratio, rooms per person, distance to major cities)
- Interactive Streamlit web app with USD / INR display
- Price range (not just one number) and "what-if" charts

## Tech stack
Python, scikit-learn, pandas, NumPy, Matplotlib, Streamlit

## Run it locally
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```
The model trains itself on first launch (about a minute).

## Files
- `HomeValue_AI.ipynb`: data exploration, model training and evaluation
- `train.py`: training script
- `features.py`: feature engineering
- `app.py`: Streamlit web app