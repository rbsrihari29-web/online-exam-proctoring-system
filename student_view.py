"""Student view: continuous webcam proctoring + live alerts + violation tracking."""

import time

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

import db
import auth
from video_processor import AGWFNVideoProcessor
from utils.model_loader import run_inference

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

STATE_COLORS = {
    "NORMAL": "#1a9c4a",
    "MALPRACTICE": "#d92d2d",
    "WARNING-NO CANDIDATE VISIBLE": "#e0a412",
    "INITIALIZING": "#888888",
    "ERROR": "#888888",
}


def render(model):
    user = auth.current_user()
    username = user["username"]

    # Re-fetch fresh user state each run in case admin changed something mid-session
    fresh = db.get_user(username)
    if fresh and fresh["locked"]:
        reason = (
            "Your account has been disallowed by the administrator after reviewing "
            "your flagged malpractice images. Contact your exam administrator."
            if fresh["disallowed"] else
            "⚠️ FINAL WARNING: You exceeded the allowed malpractice count. "
            "Your session has been ended and flagged for admin review."
        )
        auth.logout(reason=reason)
        st.rerun()

    st.title("🎓 Live Exam Proctoring")
    st.caption(f"Logged in as **{user['full_name']}** ({username})")

    col_logout, _ = st.columns([1, 5])
    with col_logout:
        if st.button("Log out"):
            auth.logout()
            st.rerun()

    settings = db.get_settings()
    max_violations = settings["max_violations"]
    current_count = fresh["violation_count"] if fresh else 0

    st.info(
        f"⚠️ **Exam rules:** Keep your face visible at all times. Cell phones, books, "
        f"laptops, headphones, and TVs/monitors are not permitted in view. "
        f"You are allowed **{max_violations - current_count}** more flagged violations "
        f"before your session is automatically ended."
    )

    col_cam, col_status = st.columns([1.4, 1])

    with col_cam:
        st.subheader("Your Camera")

        def processor_factory():
            return AGWFNVideoProcessor(username, model, run_inference)

        ctx = webrtc_streamer(
            key=f"proctor-{username}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=processor_factory,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_status:
        st.subheader("Live Status")
        status_placeholder = st.empty()
        violation_placeholder = st.empty()

        if ctx.state.playing and ctx.video_processor:
            status = ctx.video_processor.get_status()
            state = status.get("state", "INITIALIZING")
            color = STATE_COLORS.get(state, "#888888")

            status_placeholder.markdown(
                f"""
                <div style="padding:1.2rem;border-radius:0.6rem;background:{color}22;
                            border:2px solid {color};text-align:center;">
                    <span style="font-size:1.3rem;font-weight:700;color:{color};">
                        {state}
                    </span>
                    <br><span style="font-size:0.8rem;color:#555;">
                        latency: {status.get('latency_ms', 0):.0f} ms
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if status.get("probs"):
                st.bar_chart(status["probs"])

            live_count = status.get("violation_count", current_count)
            violation_placeholder.metric(
                "Flagged violations this session",
                f"{live_count} / {max_violations}",
            )

            if status.get("locked_out"):
                st.error(
                    "🚫 FINAL WARNING — malpractice limit exceeded. "
                    "Ending session and notifying admin..."
                )
                time.sleep(2)
                auth.logout(
                    reason="Your session was ended after exceeding the allowed "
                           "malpractice count. Awaiting admin review."
                )
                st.rerun()
        else:
            status_placeholder.info("Click **START** above to begin live proctoring.")
            violation_placeholder.metric(
                "Flagged violations this session",
                f"{current_count} / {max_violations}",
            )

    st.divider()
    with st.expander("Your past flagged violations (visible to admin for review)"):
        my_violations = db.get_violations(username)
        if my_violations:
            st.dataframe(
                [{"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v["timestamp"])),
                  "state": v["state"], "flagged": v["flagged_classes"]}
                 for v in my_violations],
                use_container_width=True,
            )
        else:
            st.caption("No violations recorded.")

    # Auto-refresh so the status panel updates live while streaming
    if ctx.state.playing:
        time.sleep(1.0)
        st.rerun()
