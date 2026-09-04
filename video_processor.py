"""
Continuous webcam video processor (streamlit-webrtc).

Runs in a background thread managed by streamlit-webrtc. Every N frames
(N = admin-configured frame_skip) it runs AGWFN inference on the current
frame. If the decision is MALPRACTICE, it:
  1. Saves the frame to captures/<username>/<timestamp>.jpg
  2. Logs a violation row in SQLite
  3. Increments the student's violation_count
  4. If violation_count reaches the admin-configured max, locks the account
     (checked by the main thread on next rerun, which forces logout)

Thread-safety: streamlit-webrtc calls `recv()` on a separate thread from the
Streamlit script thread, so all shared state goes through `self.status`
(protected by a lock) which the main thread polls, and through SQLite writes
(which are safe for this single-process, low-concurrency use case).
"""

import time
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from streamlit_webrtc import VideoProcessorBase

import db
from utils.model_loader import decide_alert
from utils.preprocess import frame_to_tensor


class AGWFNVideoProcessor(VideoProcessorBase):
    def __init__(self, username: str, model, run_inference_fn):
        self.username = username
        self.model = model
        self.run_inference_fn = run_inference_fn

        self.frame_count = 0
        self.lock = threading.Lock()
        self.status = {
            "state": "INITIALIZING",
            "probs": {},
            "latency_ms": 0.0,
            "last_updated": time.time(),
        }

        user_dir = Path(__file__).parent / "captures" / username
        user_dir.mkdir(parents=True, exist_ok=True)
        self.capture_dir = user_dir

    def get_status(self) -> dict:
        with self.lock:
            return dict(self.status)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        self.frame_count += 1

        settings = db.get_settings()
        frame_skip = max(1, settings["frame_skip"])

        if self.frame_count % frame_skip == 0 and self.model is not None:
            try:
                t0 = time.time()
                tensor = frame_to_tensor(img)
                probs = self.run_inference_fn(self.model, tensor)
                latency_ms = (time.time() - t0) * 1000

                result = decide_alert(probs, threshold=settings["threshold"])
                state = result["state"]

                with self.lock:
                    self.status = {
                        "state": state,
                        "probs": result["probs"],
                        "flags": result["flags"],
                        "latency_ms": latency_ms,
                        "last_updated": time.time(),
                    }

                if state == "MALPRACTICE":
                    self._handle_malpractice(img, result)

            except Exception as e:
                with self.lock:
                    self.status = {
                        "state": "ERROR",
                        "probs": {},
                        "error": str(e),
                        "last_updated": time.time(),
                    }

        # Draw a status overlay onto the outgoing video frame
        annotated = self._annotate(img)
        return av.VideoFrame.from_ndarray(annotated, format="rgb24")

    def _handle_malpractice(self, frame_rgb: np.ndarray, result: dict):
        timestamp = time.time()
        fname = f"{int(timestamp*1000)}.jpg"
        fpath = self.capture_dir / fname

        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(fpath), bgr)

        flagged = [c for c, v in result["flags"].items() if v and c != "Person"]
        db.log_violation(
            username=self.username,
            image_path=str(fpath),
            state=result["state"],
            flagged_classes=", ".join(flagged),
        )

        new_count = db.increment_violation(self.username)
        settings = db.get_settings()

        with self.lock:
            self.status["violation_count"] = new_count
            self.status["max_violations"] = settings["max_violations"]

        if new_count >= settings["max_violations"]:
            db.lock_user(self.username)
            with self.lock:
                self.status["locked_out"] = True

    def _annotate(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        with self.lock:
            state = self.status.get("state", "")
        color = {
            "NORMAL": (0, 200, 0),
            "MALPRACTICE": (220, 30, 30),
            "WARNING-NO CANDIDATE VISIBLE": (230, 165, 0),
        }.get(state, (120, 120, 120))

        h, w, _ = out.shape
        cv2.rectangle(out, (0, 0), (w, 40), color, thickness=-1)
        cv2.putText(out, state or "...", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return out
