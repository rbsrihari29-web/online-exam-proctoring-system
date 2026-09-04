"""Login form + session helpers."""

import streamlit as st
import db


def is_logged_in() -> bool:
    return st.session_state.get("logged_in", False)


def current_user() -> dict:
    return st.session_state.get("user", {})


def login(user: dict):
    st.session_state.logged_in = True
    st.session_state.user = user


def logout(reason: str = None):
    st.session_state.logged_in = False
    st.session_state.user = {}
    st.session_state.pop("webrtc_ctx_key", None)
    if reason:
        st.session_state["logout_reason"] = reason


def render_login_form():
    st.title("🎓 AGWFN Exam Proctoring")
    st.caption("Sign in to continue")

    if st.session_state.get("logout_reason"):
        st.error(st.session_state.pop("logout_reason"))

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", use_container_width=True)

    if submitted:
        user = db.verify_login(username.strip(), password)
        if user is None:
            st.error("Invalid username or password.")
        elif user["role"] == "student" and user["locked"]:
            if user["disallowed"]:
                st.error(
                    "Your account has been disallowed by the admin after review. "
                    "Contact your exam administrator."
                )
            else:
                st.error(
                    "Your account is locked pending admin review of flagged malpractice. "
                    "Please wait for the administrator to review your case."
                )
        else:
            login(user)
            st.rerun()

    with st.expander("Demo accounts"):
        st.code(
            "admin / admin123\n"
            "student1 / student123\n"
            "student2 / student123",
            language="text",
        )
