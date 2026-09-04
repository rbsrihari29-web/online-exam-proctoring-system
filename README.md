# AGWFN Live Exam Proctoring Dashboard

Login-gated Streamlit app with continuous webcam proctoring, automatic
malpractice capture, and an admin review workflow — built on your AGWFN
4-branch ensemble (ViT-Small, EfficientNet-B2, ResNet50, MobileNetV2).

## Features

**Students**
- Continuous live webcam stream (via `streamlit-webrtc`) with real-time AGWFN inference
- On-screen alert overlay: `NORMAL` / `MALPRACTICE` / `WARNING-NO CANDIDATE VISIBLE`
- Live violation counter (`X / max allowed`)
- Auto-logout with a final warning once the admin-set violation limit is hit
- Can view their own past flagged violations

**Admins**
- Set decision threshold, inference frame-skip rate, and max allowed malpractice count
- Live student monitor table (status + violation counts) and full violation history
- Pending Reviews: see captured malpractice images per locked-out student, then
  **Allow** (resets count, unlocks) or **Disallow** (permanently blocks) after review
- Create new student and admin accounts

Every `MALPRACTICE` detection saves the actual webcam frame to
`captures/<username>/<timestamp>.jpg` and logs a row in SQLite — that's what
the admin reviews before deciding.

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`streamlit-webrtc` needs `av` (installed above) and a working webcam/browser
permission. On first run your browser will ask for camera access.

## 2. Add your trained weights

Copy your trained checkpoint to:

```
models/ensemble_best.pth
```

(This matches the `AttentionFusionEnsemble` checkpoint format already wired
into `utils/model_loader.py`.)

## 3. Run it

```bash
streamlit run app.py
```

A SQLite DB (`data/agwfn.db`) is created automatically on first run, seeded
with demo accounts:

| Username  | Password    | Role    |
|-----------|-------------|---------|
| admin     | admin123    | admin   |
| student1  | student123  | student |
| student2  | student123  | student |

Change these in `db.py` (`_seed_demo_accounts`) before any real deployment,
and swap the sha256 hashing for bcrypt/argon2 — this project uses sha256 for
simplicity since it's a research/demo dashboard, not a production auth system.

## 4. How the lockout flow works

1. Admin sets **max malpractice count** (Detection Settings tab).
2. A student's webcam stream runs continuously; every `frame_skip` frames,
   AGWFN runs inference.
3. Each `MALPRACTICE` result saves the frame + increments that student's count.
4. When the count reaches the admin's limit: the student's account is locked,
   they see a final warning, and are automatically logged out.
5. The admin's **Pending Reviews** tab lights up with that student, showing
   their captured frames.
6. Admin clicks **Allow** (resets the count, unlocks them to log back in) or
   **Disallow** (permanently blocks further login).

## 5. Push to GitHub (Git LFS for the weights file)

```bash
git init
git lfs install
git lfs track "*.pt" "*.pth"     # already set up in .gitattributes
git add .
git commit -m "Add AGWFN login-gated live proctoring dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/agwfn-exam-proctoring.git
git push -u origin main
```

`captures/` and `data/agwfn.db` are gitignored — captured images and the
student database are local/runtime state, not something you want committed.

## Files

```
agwfn-dashboard/
├── app.py               # entry point: login gate + role routing
├── auth.py              # login form + session helpers
├── db.py                # SQLite: users, violations, settings
├── video_processor.py   # streamlit-webrtc processor (continuous inference + capture)
├── student_view.py      # student UI: live stream, alerts, violation counter
├── admin_view.py        # admin UI: settings, monitor, review, add accounts
├── utils/
│   ├── model_loader.py  # AGWFN AttentionFusionEnsemble loader (exact match to training)
│   └── preprocess.py    # webcam frame -> tensor
├── models/              # put ensemble_best.pth here
├── captures/            # auto-created; malpractice frames saved here per student
├── data/                # auto-created; agwfn.db (SQLite)
├── requirements.txt
├── .gitattributes       # Git LFS tracking for *.pt/*.pth
└── .gitignore
```
## 2. Download trained weights

Download ensemble_best.pth from Google Drive:

https://drive.google.com/file/d/1j53hFdPRBku2z7xiIQqnrqpGc8Io_iVT/view?usp=drive_link

Or via command line:

```bash
pip install gdown
gdown https://drive.google.com/file/d/1j53hFdPRBku2z7xiIQqnrqpGc8Io_iVT/view?usp=drive_link
```

Place the downloaded file at:

```
models/ensemble_best.pth
```
