import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, "data", "Sleep_Health_Lifestyle_Dataset.xlsx")
MODEL_FILE = os.path.join(BASE_DIR, "sleep_model.pkl")

TARGET_MAPPING = {
    "No Sleep Disorder": 0,
    "Insomnia": 1,
    "Sleep Apnea": 2,
}
TARGET_MAPPING_INV = {value: key for key, value in TARGET_MAPPING.items()}

RAW_INPUT_FIELDS = [
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Daily Steps",
    "Systolic BP",
    "Diastolic BP",
    "Heart Rate",
]

ENGINEERED_FEATURES = [
    "Pulse Pressure",
    "Sleep Quality Index",
]

BEST_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 0.7,
    "colsample_bytree": 1.0,
}

VALIDATION_SCHEMA = {
    "gender": {"type": "select", "label": "Gender", "options": ["Male", "Female"]},
    "age": {"type": "number", "label": "Age", "unit": "years", "min": 18, "max": 80, "step": 1},
    "occupation": {"type": "select", "label": "Occupation", "options": [
        "Accountant", "Doctor", "Engineer (Non-Software)", "Entrepreneur",
        "Lawyer", "Manager", "Nurse", "Sales Representative", "Scientist",
        "Software Engineer", "Student", "Teacher"
    ]},
    "sleepDuration": {"type": "number", "label": "Sleep Duration", "unit": "hours/night", "min": 4, "max": 10, "step": 0.1},
    "sleepQuality": {"type": "number", "label": "Quality of Sleep", "unit": "/10", "min": 1, "max": 10, "step": 1},
    "physicalActivity": {"type": "number", "label": "Physical Activity", "unit": "minutes/day", "min": 0, "max": 120, "step": 5},
    "stressLevel": {"type": "number", "label": "Stress Level", "unit": "/10", "min": 1, "max": 10, "step": 1},
    "bmiCategory": {"type": "select", "label": "BMI Category", "options": ["Underweight", "Normal", "Overweight", "Obese"]},
    "dailySteps": {"type": "number", "label": "Daily Steps", "unit": "steps/day", "min": 1000, "max": 20000, "step": 100},
    "systolicBP": {"type": "number", "label": "Systolic BP", "unit": "mmHg", "min": 90, "max": 180, "step": 1},
    "diastolicBP": {"type": "number", "label": "Diastolic BP", "unit": "mmHg", "min": 60, "max": 120, "step": 1},
    "heartRate": {"type": "number", "label": "Resting Heart Rate", "unit": "bpm", "min": 50, "max": 120, "step": 1},
}


def load_and_clean_data():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    df = pd.read_excel(DATASET_FILE, keep_default_na=False)

    if "Person ID" in df.columns:
        df = df.drop(columns=["Person ID"])

    df = df.drop_duplicates().copy()

    text_columns = [
        "Gender",
        "Occupation",
        "BMI Category",
        "Blood Pressure",
        "Sleep Disorder",
    ]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    df["Gender"] = df["Gender"].str.title()
    df["Occupation"] = df["Occupation"].str.title()
    df["BMI Category"] = df["BMI Category"].str.title()
    df["BMI Category"] = df["BMI Category"].replace({"Over Weight": "Overweight"})

    df = df.replace("", np.nan)

    numeric_columns = [
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    blood_pressure = df["Blood Pressure"].astype("string").str.split("/", expand=True)
    if blood_pressure.shape[1] != 2:
        raise ValueError("Blood Pressure must contain values in systolic/diastolic format.")

    df["Systolic BP"] = pd.to_numeric(blood_pressure[0], errors="coerce")
    df["Diastolic BP"] = pd.to_numeric(blood_pressure[1], errors="coerce")
    df = df.drop(columns=["Blood Pressure"])

    df["Sleep Disorder"] = df["Sleep Disorder"].replace({"None": "No Sleep Disorder"})

    required_columns = set(RAW_INPUT_FIELDS) | {"Sleep Disorder"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required column(s): " + ", ".join(missing_columns)
        )

    return df


def engineer_features(df):
    df = df.copy()

    required = {
        "Systolic BP",
        "Diastolic BP",
        "Sleep Duration",
        "Quality of Sleep",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Cannot engineer model features; missing column(s): " + ", ".join(missing)
        )

    df["Pulse Pressure"] = df["Systolic BP"] - df["Diastolic BP"]
    df["Sleep Quality Index"] = (
        (df["Sleep Duration"] / 24) + (df["Quality of Sleep"] / 10)
    ) / 2
    df["Sleep Quality Index"] = df["Sleep Quality Index"].round(3)

    return df


def build_preprocessor(numerical_features, categorical_features):
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_model_pipeline(numerical_features, categorical_features):
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        **BEST_XGB_PARAMS,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def build_class_profiles(df, numerical_features):
    profiles = {}
    for label in TARGET_MAPPING:
        subset = df.loc[df["Sleep Disorder"] == label, numerical_features]
        profiles[label] = subset.mean(numeric_only=True).round(2).to_dict()
    return profiles


def train_and_save_model():
    df = engineer_features(load_and_clean_data())

    X = df.drop(columns=["Sleep Disorder"])
    y = df["Sleep Disorder"]

    unknown_labels = sorted(set(y.dropna()) - set(TARGET_MAPPING))
    if unknown_labels:
        raise ValueError("Unknown target label(s): " + ", ".join(map(str, unknown_labels)))

    y_encoded = y.map(TARGET_MAPPING)
    if y_encoded.isna().any():
        raise ValueError("Target contains missing or unmapped values.")

    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    missing_engineered = [
        feature for feature in ENGINEERED_FEATURES if feature not in X.columns
    ]
    if missing_engineered:
        raise ValueError(
            "Engineered feature(s) missing: " + ", ".join(missing_engineered)
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    validation_pipeline = build_model_pipeline(
        numerical_features,
        categorical_features,
    )
    validation_pipeline.fit(X_train, y_train)

    y_pred = validation_pipeline.predict(X_test)
    y_proba = validation_pipeline.predict_proba(X_test)

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_proba, multi_class="ovr"),
    }

    final_pipeline = build_model_pipeline(
        numerical_features,
        categorical_features,
    )
    final_pipeline.fit(X, y_encoded)

    class_profiles = build_class_profiles(df, numerical_features)

    artifact = {
        "pipeline": final_pipeline,
        "target_mapping": TARGET_MAPPING,
        "target_mapping_inv": TARGET_MAPPING_INV,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "feature_order": X.columns.tolist(),
        "raw_input_fields": RAW_INPUT_FIELDS,
        "engineered_features": ENGINEERED_FEATURES,
        "class_profiles": class_profiles,
        "test_metrics": test_metrics,
        "model_params": BEST_XGB_PARAMS,
        "validation_schema": VALIDATION_SCHEMA,
        "artifact_version": 4,
    }

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Model artifact saved to '{MODEL_FILE}'.")
    return artifact


if __name__ == "__main__":
    train_and_save_model()
