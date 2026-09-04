"""Admin view: configure detection settings, manage students, review malpractice
captures, and allow/disallow locked students."""

import time
from pathlib import Path

import streamlit as st

import db
import auth


def render():
    user = auth.current_user()
    st.title("🛠️ Admin Dashboard")
    st.caption(f"Logged in as **{user['full_name']}** ({user['username']})")

    col_logout, _ = st.columns([1, 5])
    with col_logout:
        if st.button("Log out"):
            auth.logout()
            st.rerun()

    pending = db.get_pending_reviews()
    if pending:
        st.error(
            f"🔔 **{len(pending)} student(s) awaiting review** — "
            f"they exceeded the malpractice limit and have been auto-logged-out. "
            f"See the 'Pending Reviews' tab below."
        )

    tab_settings, tab_students, tab_review, tab_add = st.tabs(
        ["Detection Settings", "Student Monitor", "Pending Reviews", "Add Student"]
    )

    with tab_settings:
        render_settings()

    with tab_students:
        render_student_monitor()

    with tab_review:
        render_pending_reviews(pending)

    with tab_add:
        render_add_student()


def render_settings():
    st.subheader("Detection Settings")
    settings = db.get_settings()

    with st.form("settings_form"):
        threshold = st.slider(
            "Decision threshold", 0.0, 1.0, float(settings["threshold"]), 0.01,
            help="Minimum class probability to count as detected.",
        )
        frame_skip = st.slider(
            "Run inference every N frames", 1, 60, int(settings["frame_skip"]),
            help="Higher = lighter on CPU/GPU, lower = more responsive detection.",
        )
        max_violations = st.slider(
            "Malpractice count allowed before lockout", 1, 20, int(settings["max_violations"]),
            help="Once a student's flagged violations reach this number, they are "
                 "auto-logged-out and flagged for your review.",
        )
        submitted = st.form_submit_button("Save settings", use_container_width=True)

    if submitted:
        db.update_settings(threshold, frame_skip, max_violations)
        st.success("Settings updated. New values apply immediately to live sessions.")


def render_student_monitor():
    st.subheader("Student Monitor")
    students = db.list_students()

    if not students:
        st.caption("No students registered yet.")
        return

    settings = db.get_settings()
    rows = []
    for s in students:
        status = "🔴 Disallowed" if s["disallowed"] else \
                 "🟠 Locked (pending review)" if s["locked"] else "🟢 Active"
        rows.append({
            "Username": s["username"],
            "Name": s["full_name"],
            "Status": status,
            "Violations": f"{s['violation_count']} / {settings['max_violations']}",
        })
    st.dataframe(rows, use_container_width=True)

    st.divider()
    st.subheader("Violation History (all students)")
    all_violations = db.get_violations()
    if all_violations:
        st.dataframe(
            [{"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v["timestamp"])),
              "student": v["username"], "state": v["state"],
              "flagged": v["flagged_classes"], "reviewed": bool(v["reviewed"])}
             for v in all_violations],
            use_container_width=True,
            height=300,
        )
    else:
        st.caption("No violations logged yet.")


def render_pending_reviews(pending):
    st.subheader("Pending Reviews")
    st.caption(
        "Students below exceeded the malpractice limit and were automatically "
        "logged out. Review their captured images, then allow or disallow them."
    )

    if not pending:
        st.success("No students currently awaiting review.")
        return

    for s in pending:
        with st.container(border=True):
            st.markdown(f"### {s['full_name']} ({s['username']})")
            st.caption(f"Total flagged violations: {s['violation_count']}")

            violations = db.get_violations(s["username"])
            if violations:
                st.write("**Captured malpractice frames:**")
                cols = st.columns(4)
                for i, v in enumerate(violations[:8]):
                    img_path = Path(v["image_path"])
                    with cols[i % 4]:
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                            st.caption(
                                f"{time.strftime('%H:%M:%S', time.localtime(v['timestamp']))} "
                                f"— {v['flagged_classes'] or v['state']}"
                            )
                        else:
                            st.caption("Image file missing")
            else:
                st.caption("No captured images found.")

            col_allow, col_disallow = st.columns(2)
            with col_allow:
                if st.button(f"✅ Allow {s['username']}", key=f"allow_{s['username']}",
                             use_container_width=True):
                    db.set_user_decision(s["username"], allow=True)
                    st.success(f"{s['username']} has been allowed to resume. Violation count reset.")
                    st.rerun()
            with col_disallow:
                if st.button(f"🚫 Disallow {s['username']}", key=f"disallow_{s['username']}",
                             use_container_width=True, type="primary"):
                    db.set_user_decision(s["username"], allow=False)
                    st.warning(f"{s['username']} has been disallowed from further access.")
                    st.rerun()


def render_add_student():
    st.subheader("Add Student Account")
    with st.form("add_student_form"):
        username = st.text_input("Username")
        full_name = st.text_input("Full name")
        password = st.text_input("Temporary password", type="password")
        submitted = st.form_submit_button("Create student account", use_container_width=True)

    if submitted:
        if not username or not password:
            st.error("Username and password are required.")
        elif db.create_user(username.strip(), password, full_name.strip(), role="student"):
            st.success(f"Student account '{username}' created.")
        else:
            st.error(f"Username '{username}' already exists.")

    st.divider()
    st.subheader("Add Admin Account")
    with st.form("add_admin_form"):
        a_username = st.text_input("Admin username")
        a_full_name = st.text_input("Admin full name")
        a_password = st.text_input("Admin password", type="password")
        a_submitted = st.form_submit_button("Create admin account", use_container_width=True)

    if a_submitted:
        if not a_username or not a_password:
            st.error("Username and password are required.")
        elif db.create_user(a_username.strip(), a_password, a_full_name.strip(), role="admin"):
            st.success(f"Admin account '{a_username}' created.")
        else:
            st.error(f"Username '{a_username}' already exists.")
