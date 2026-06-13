"""Página 6: Predicción con modelo entrenado."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import tempfile

import pandas as pd
import streamlit as st
import tensorflow as tf

from components.ui_helpers import setup_branding
from utils.feature_meta import (
    compute_feature_meta,
    load_ann_assets,
    load_rf_bundle,
    predict_ann,
    predict_rf,
)

st.set_page_config(
    page_title="Predicción",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_branding(
    page_title="Predicción",
    page_subtitle="Inferencia con modelos entrenados y rangos de referencia del dataset",
    show_logo=True,
)

SOURCE_LABELS = {
    "rf_session": "Random Forest (Pipeline Estándar)",
    "ann_session": "Red Neuronal ANN (Análisis Avanzado)",
    "upload": "Subir modelo entrenado",
}


def _available_sources() -> list[str]:
    sources: list[str] = []
    if "ml_model" in st.session_state:
        sources.append("rf_session")
    if "clf" in st.session_state:
        sources.append("ann_session")
    sources.append("upload")
    return sources


def _load_uploaded_rf(uploaded_file) -> None:
    bundle = load_rf_bundle(uploaded_file.getvalue())
    st.session_state["pred_uploaded_kind"] = "rf"
    st.session_state["pred_uploaded_model"] = bundle["model"]
    st.session_state["pred_uploaded_meta"] = bundle
    st.session_state["pred_uploaded_name"] = uploaded_file.name


def _load_uploaded_ann(keras_file, assets_file) -> None:
    assets = load_ann_assets(assets_file.getvalue())

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(keras_file.getvalue())
        tmp_path = tmp.name

    try:
        model = tf.keras.models.load_model(tmp_path)
    finally:
        os.unlink(tmp_path)

    st.session_state["pred_uploaded_kind"] = "ann"
    st.session_state["pred_uploaded_model"] = model
    st.session_state["pred_uploaded_meta"] = assets
    st.session_state["pred_uploaded_name"] = f"{keras_file.name} + {assets_file.name}"


def _resolve_rf_context(model, feature_meta: dict, extra: dict | None = None) -> dict:
    extra = extra or {}
    feature_names = list(
        extra.get("feature_names")
        or getattr(model, "feature_names_in_", list(feature_meta.keys()))
    )
    if not feature_meta and st.session_state.get("clean_data") is not None:
        feature_meta = compute_feature_meta(st.session_state["clean_data"], feature_names)

    return {
        "kind": "rf",
        "model": model,
        "feature_names": feature_names,
        "feature_meta": feature_meta,
        "form_features": feature_names,
        "task": extra.get("task", st.session_state.get("ml_task", "Clasificación")),
        "target": extra.get("target", st.session_state.get("ml_target")),
        "label": extra.get("label", "Random Forest"),
    }


def _resolve_ann_context(model, scaler, label_encoder, processed_features, feature_columns, feature_meta, extra: dict | None = None) -> dict:
    extra = extra or {}
    if not feature_columns and feature_meta:
        feature_columns = list(feature_meta.keys())
    if not feature_meta and feature_columns:
        clf = st.session_state.get("clf")
        if clf is not None and hasattr(clf, "df"):
            feature_meta = compute_feature_meta(clf.df, feature_columns)

    return {
        "kind": "ann",
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "processed_features": processed_features,
        "feature_columns": feature_columns,
        "feature_meta": feature_meta,
        "form_features": feature_columns,
        "target": extra.get("target", st.session_state.get("ann_target")),
        "label": extra.get("label", "Red Neuronal ANN"),
    }


def _resolve_context(source: str) -> dict | None:
    if source == "rf_session":
        return _resolve_rf_context(
            st.session_state["ml_model"],
            st.session_state.get("ml_feature_meta", {}),
            {
                "feature_names": st.session_state.get("ml_feature_names"),
                "task": st.session_state.get("ml_task"),
                "target": st.session_state.get("ml_target"),
                "label": "Random Forest (sesión)",
            },
        )

    if source == "ann_session":
        clf = st.session_state["clf"]
        return _resolve_ann_context(
            clf.model,
            clf.scaler,
            clf.label_encoder,
            clf.processed_features,
            clf.feature_columns,
            st.session_state.get("ann_feature_meta", {}),
            {"target": st.session_state.get("ann_target", clf.target_column), "label": "ANN (sesión)"},
        )

    if source == "upload":
        kind = st.session_state.get("pred_uploaded_kind")
        if kind == "rf" and st.session_state.get("pred_uploaded_model") is not None:
            meta = st.session_state.get("pred_uploaded_meta", {})
            return _resolve_rf_context(
                st.session_state["pred_uploaded_model"],
                meta.get("feature_meta", {}),
                {
                    "feature_names": meta.get("feature_names"),
                    "task": meta.get("task"),
                    "target": meta.get("target"),
                    "label": f"RF cargado ({st.session_state.get('pred_uploaded_name', 'archivo')})",
                },
            )

        if kind == "ann" and st.session_state.get("pred_uploaded_model") is not None:
            meta = st.session_state.get("pred_uploaded_meta", {})
            return _resolve_ann_context(
                st.session_state["pred_uploaded_model"],
                meta["scaler"],
                meta.get("label_encoder"),
                meta["processed_features"],
                meta.get("feature_columns", []),
                meta.get("feature_meta", {}),
                {
                    "target": meta.get("target_column"),
                    "label": f"ANN cargado ({st.session_state.get('pred_uploaded_name', 'archivo')})",
                },
            )

    return None


def _render_input_form(context: dict) -> dict | None:
    feature_meta = context.get("feature_meta", {})
    form_features = context.get("form_features", [])

    if not form_features:
        st.error("No se encontraron variables de entrada para este modelo.")
        return None

    st.markdown("### Variables de entrada")
    st.caption(
        "Ingresa un valor por variable. Los rangos mostrados provienen del dataset de entrenamiento."
    )

    inputs: dict = {}
    with st.form("prediction_form", clear_on_submit=False):
        cols = st.columns(2)
        for idx, feature in enumerate(form_features):
            meta = feature_meta.get(feature, {})
            with cols[idx % 2]:
                if meta.get("dtype") == "categorical":
                    values = meta.get("values", [])
                    if not values:
                        values = [""]
                    inputs[feature] = st.selectbox(
                        feature,
                        options=values,
                        help=f"Valores vistos en entrenamiento: {', '.join(values)}",
                    )
                else:
                    default = float(meta.get("median", 0.0))
                    min_val = meta.get("min")
                    max_val = meta.get("max")
                    caption = None
                    if min_val is not None and max_val is not None:
                        caption = f"Rango entrenamiento: {min_val:g} – {max_val:g}"

                    inputs[feature] = st.number_input(
                        feature,
                        value=default,
                        format="%.6f",
                        help=caption,
                    )
                    if caption:
                        st.caption(caption)

        submitted = st.form_submit_button("Predecir", type="primary", use_container_width=True)

    return inputs if submitted else None


def _display_prediction(context: dict, result: dict) -> None:
    st.divider()
    st.markdown("### Resultado de la predicción")

    task = context.get("task", "Clasificación")
    target = context.get("target")

    if context["kind"] == "ann" and "prediction_label" in result:
        st.success(f"**Predicción:** {result['prediction_label']}")
    elif task == "Regresión":
        st.success(f"**Valor predicho{' de ' + target if target else ''}:** {result['prediction']:.4f}")
    else:
        st.success(f"**Clase predicha{' (' + target + ')' if target else ''}:** {result['prediction']}")

    if "confidence" in result:
        st.metric("Confianza", f"{result['confidence']:.1%}")

    if "probability" in result:
        st.metric("Probabilidad clase positiva", f"{result['probability']:.1%}")

    if result.get("probabilities"):
        st.markdown("**Distribución de probabilidades**")
        prob_df = pd.DataFrame(
            [{"Clase": cls, "Probabilidad": prob} for cls, prob in result["probabilities"].items()]
        )
        st.dataframe(prob_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.header("6. Predicción con Modelo Entrenado")
    st.markdown(
        """
        Usa un modelo ya entrenado para predecir un nuevo registro.
        Puedes elegir un modelo de la sesión actual o cargar uno exportado previamente.
        """
    )

    sources = _available_sources()
    default_source = sources[0]

    st.subheader("Seleccionar modelo")
    source = st.radio(
        "Fuente del modelo",
        options=sources,
        format_func=lambda key: SOURCE_LABELS[key],
        index=sources.index(default_source),
        horizontal=True,
    )

    if source == "upload":
        st.markdown("#### Cargar modelo exportado")
        upload_kind = st.radio(
            "Tipo de archivo",
            options=["rf", "ann"],
            format_func=lambda k: "Random Forest (.pkl)" if k == "rf" else "Red Neuronal ANN (.keras + assets)",
            horizontal=True,
        )

        if upload_kind == "rf":
            uploaded_pkl = st.file_uploader(
                "Archivo del modelo (.pkl)",
                type=["pkl"],
                key="pred_rf_upload",
            )
            if uploaded_pkl is not None:
                try:
                    _load_uploaded_rf(uploaded_pkl)
                    st.success(f"Modelo cargado: {uploaded_pkl.name}")
                except Exception as exc:
                    st.error(f"No se pudo cargar el modelo: {exc}")
        else:
            col_k, col_a = st.columns(2)
            with col_k:
                uploaded_keras = st.file_uploader(
                    "Modelo Keras (.keras)",
                    type=["keras", "h5"],
                    key="pred_ann_keras_upload",
                )
            with col_a:
                uploaded_assets = st.file_uploader(
                    "Assets auxiliares (_assets.pkl)",
                    type=["pkl"],
                    key="pred_ann_assets_upload",
                )

            if uploaded_keras is not None and uploaded_assets is not None:
                try:
                    _load_uploaded_ann(uploaded_keras, uploaded_assets)
                    st.success("Modelo ANN cargado correctamente.")
                except Exception as exc:
                    st.error(f"No se pudo cargar el modelo ANN: {exc}")

        if st.session_state.get("pred_uploaded_kind"):
            if st.button("Limpiar modelo cargado"):
                for key in [
                    "pred_uploaded_kind",
                    "pred_uploaded_model",
                    "pred_uploaded_meta",
                    "pred_uploaded_name",
                ]:
                    st.session_state.pop(key, None)
                st.rerun()

    context = _resolve_context(source)

    if context is None:
        if source == "upload":
            st.info(
                "Sube un archivo `.pkl` (Random Forest) o un par `.keras` + `_assets.pkl` (ANN) para continuar."
            )
        else:
            st.warning(
                "No hay un modelo entrenado en la sesión. "
                "Entrena uno en **Machine Learning** o **Análisis Avanzado**, o sube un modelo exportado."
            )
        return

    info_cols = st.columns(3)
    info_cols[0].metric("Modelo activo", context.get("label", context["kind"].upper()))
    if context.get("target"):
        info_cols[1].metric("Variable objetivo", context["target"])
    info_cols[2].metric("Variables de entrada", len(context.get("form_features", [])))

    inputs = _render_input_form(context)
    if inputs is None:
        return

    try:
        if context["kind"] == "rf":
            result = predict_rf(context["model"], inputs, context["feature_names"])
        else:
            result = predict_ann(
                context["model"],
                inputs,
                context["feature_columns"],
                context["processed_features"],
                context["scaler"],
                context.get("label_encoder"),
                context.get("feature_meta"),
            )
        _display_prediction(context, result)
    except Exception as exc:
        st.error(f"Error al generar la predicción: {exc}")


if __name__ == "__main__":
    main()
