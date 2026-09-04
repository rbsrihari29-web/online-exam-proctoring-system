"""
AGWFN Live Exam Proctoring — Main App
======================================
Login-gated, role-based Streamlit app:
  - students: continuous webcam proctoring, live malpractice alerts,
              automatic violation logging + lockout
  - admins:   detection settings, student monitor, violation review,
              allow/disallow after reviewing captured images, account creation

Run:
    streamlit run app.py

First-time setup: put your trained checkpoint at models/ensemble_best.pth
(or use the sidebar uploader). Demo accounts are seeded automatically —
see the login screen for credentials.
"""

from pathlib import Path

import streamlit as st

import db
import auth
import student_view
import admin_view
from utils.model_loader import load_checkpoint

st.set_page_config(page_title="AGWFN Live Proctoring", page_icon="🎓", layout="wide")

db.init_db()

MODEL_DIR = Path(__file__).parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "ensemble_best.pth"


@st.cache_resource(show_spinner="Loading AGWFN ensemble...")
def get_model(path_str: str):
    return load_checkpoint(path_str, device="cpu")


def load_model_with_sidebar():
    """Model loading UI shown in the sidebar for both roles (admin can also
    swap checkpoints; students just need it loaded silently)."""
    model = None
    model_error = None

    is_admin = auth.current_user().get("role") == "admin"

    if is_admin:
        st.sidebar.title("AGWFN Model")
        model_source = st.sidebar.radio(
            "Checkpoint source",
            ["Use models/ensemble_best.pth", "Upload a checkpoint"],
        )
        uploaded_ckpt = None
        if model_source == "Upload a checkpoint":
            uploaded_ckpt = st.sidebar.file_uploader("Upload .pt / .pth", type=["pt", "pth"])
    else:
        model_source = "Use models/ensemble_best.pth"
        uploaded_ckpt = None

    try:
        if uploaded_ckpt is not None:
            tmp_path = MODEL_DIR / "uploaded_checkpoint.pth"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_ckpt.getbuffer())
            model = get_model(str(tmp_path))
        elif DEFAULT_MODEL_PATH.exists():
            model = get_model(str(DEFAULT_MODEL_PATH))
        else:
            model_error = (
                f"No checkpoint found at {DEFAULT_MODEL_PATH}. "
                "Drop your trained ensemble_best.pth there."
            )
    except Exception as e:
        model_error = f"Failed to load checkpoint: {e}"

    if is_admin:
        if model_error:
            st.sidebar.error(model_error)
        else:
            st.sidebar.success("Model loaded ✓")

    return model, model_error


def main():
    if not auth.is_logged_in():
        auth.render_login_form()
        return

    user = auth.current_user()
    model, model_error = load_model_with_sidebar()

    if user["role"] == "student":
        if model is None:
            st.error(model_error or "Model not available. Contact your administrator.")
            return
        student_view.render(model)
    elif user["role"] == "admin":
        admin_view.render()
    else:
        st.error("Unknown role.")


if __name__ == "__main__":
    main()
