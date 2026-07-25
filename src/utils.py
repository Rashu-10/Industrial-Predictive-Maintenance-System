import streamlit as st
import joblib
from config.config import MODEL_PATH, SCALER_PATH, ENCODER_PATH

def model_exists():
    return MODEL_PATH.exists()

@st.cache_resource
def load_model():
    """
    Load and cache the trained machine learning model.
    """
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    """
    Load and cache the fitted scaler.
    """
    return joblib.load(SCALER_PATH)

@st.cache_resource
def load_encoder():
    """
    Load and cache the fitted label encoder.
    """
    return joblib.load(ENCODER_PATH)

@st.cache_resource
def load_explainer():
    """
    Load and cache the model explainer.
    """
    model = load_model()
    from src.explainability import ModelExplainer
    return ModelExplainer(model)

def load_object(file_path):
    """
    Load an object from a file (non-cached).
    """
    return joblib.load(file_path)


def save_object(file_path, obj):
    """
    Save an object to a file.
    """
    joblib.dump(obj, file_path)