# Deployment

## Render

This project is a Flask application and is configured for Render.

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Health check: `/health`

The deployed service loads `sleep_model.pkl`; it does not retrain the model during startup.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Model retraining

If the model is retrained, run:

```bash
python model.py
```

This regenerates `sleep_model.pkl` from `data/Sleep_Health_Lifestyle_Dataset.xlsx`.
