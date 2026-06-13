"""Helpers to compute and use per-variable metadata for prediction forms."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def compute_feature_meta(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build metadata for each input variable from a training DataFrame.

    Numeric columns get min, max, median and dtype.
    Categorical columns get their unique training values and dtype.
    """
    if feature_columns is None:
        feature_columns = df.columns.tolist()

    meta: Dict[str, Dict[str, Any]] = {}
    for col in feature_columns:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        if pd.api.types.is_numeric_dtype(series):
            meta[col] = {
                "dtype": "numeric",
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
            }
        else:
            unique_vals = sorted(series.unique(), key=str)
            meta[col] = {
                "dtype": "categorical",
                "values": [str(v) for v in unique_vals],
            }

    return meta


def predict_rf(
    model: Any,
    inputs: Dict[str, Any],
    feature_names: List[str],
) -> Dict[str, Any]:
    """Run a Random Forest prediction from a single-row input dict."""
    row = pd.DataFrame([inputs])
    row = row.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(row)[0]

    result: Dict[str, Any] = {"prediction": prediction}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
        classes = getattr(model, "classes_", None)
        if classes is not None:
            result["probabilities"] = {
                str(cls): float(p) for cls, p in zip(classes, proba)
            }
            result["confidence"] = float(np.max(proba))

    return result


def predict_ann(
    model: Any,
    inputs: Dict[str, Any],
    feature_columns: List[str],
    processed_features: List[str],
    scaler: Any,
    label_encoder: Any = None,
    feature_meta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run an ANN prediction replicating the training preprocessing pipeline."""
    row = pd.DataFrame([inputs])

    cat_cols = []
    if feature_meta:
        cat_cols = [
            col
            for col in feature_columns
            if feature_meta.get(col, {}).get("dtype") == "categorical"
        ]
    else:
        cat_cols = row.select_dtypes(include=["object", "category"]).columns.tolist()

    if cat_cols:
        row = pd.get_dummies(row, columns=cat_cols, drop_first=True)

    row = row.reindex(columns=processed_features, fill_value=0)
    scaled = scaler.transform(row)

    y_prob = model.predict(scaled, verbose=0)
    prob = float(y_prob.flatten()[0])
    pred_class = int(prob > 0.5)

    result: Dict[str, Any] = {
        "prediction": pred_class,
        "probability": prob,
        "confidence": prob if pred_class == 1 else 1.0 - prob,
    }

    if label_encoder is not None:
        try:
            result["prediction_label"] = label_encoder.inverse_transform([pred_class])[0]
            class_labels = label_encoder.classes_
            result["probabilities"] = {
                str(class_labels[0]): 1.0 - prob,
                str(class_labels[1]): prob,
            }
        except Exception:
            pass

    return result


def load_rf_bundle(data: bytes) -> Dict[str, Any]:
    """Load a Random Forest bundle or bare estimator from pickle bytes."""
    import pickle

    obj = pickle.loads(data)

    if isinstance(obj, dict) and "model" in obj:
        return {
            "kind": obj.get("kind", "rf"),
            "model": obj["model"],
            "feature_meta": obj.get("feature_meta", {}),
            "task": obj.get("task", "Clasificación"),
            "feature_names": list(
                getattr(obj["model"], "feature_names_in_", obj.get("feature_meta", {}).keys())
            ),
        }

    model = obj
    feature_names = list(getattr(model, "feature_names_in_", []))
    return {
        "kind": "rf",
        "model": model,
        "feature_meta": {},
        "task": "Clasificación",
        "feature_names": feature_names,
    }


def load_ann_assets(data: bytes) -> Dict[str, Any]:
    """Load ANN auxiliary assets from pickle bytes."""
    import pickle

    assets = pickle.loads(data)
    return {
        "kind": "ann",
        "scaler": assets["scaler"],
        "label_encoder": assets.get("label_encoder"),
        "processed_features": assets["features"],
        "feature_columns": assets.get("feature_columns", []),
        "feature_meta": assets.get("feature_meta", {}),
    }
