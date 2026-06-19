# ⚽ Football Celebration Detector

Real-time body pose recognition that triggers football player celebration images — running entirely on CPU.

Cross your arms → Mbappé appears. Point to the sky → Messi vibes. No GPU needed.

## What Is This?

A lightweight computer vision app that watches your body through a webcam and shows a matching football celebration image beside you — in real time.

Built as a beginner-friendly introduction to:
- Real-time video processing
- Body pose & hand landmark detection with MediaPipe
- Gesture classification logic
- Image overlay rendering

No neural network training. No GPU required. Just Python + a webcam.

## Project Structure

```
football-celebration/
│
├── main.py                   ← Entry point, main loop
├── camera.py                 ← Webcam capture + mirror mode
├── body_tracker.py           ← MediaPipe Pose + Hands detection
├── celebration_detector.py   ← Celebration classification logic
├── overlay.py                ← Player image loading + side-by-side rendering
│
├── mbappe.webp               ← Arms crossed celebration
├── messi.webp                ← Pointing to the sky
├── ronaldo.jpeg              ← SIUUU power pose
├── sonnn.jpeg                ← Camera frame gesture
├── diazz.jpg                 ← Open palms celebration
├── naymar.jpg                ← Hands near ears (playful)
├── yamal.jpeg                ← Arms crossed (alternative)
│
├── screenshots/              ← Auto-created when you press S
├── requirements.txt
└── README.md
```

## Celebrations

| Gesture | Player | Description |
|---------|--------|-------------|
|  Arms crossed on chest | **Mbappé** | Cross both arms over your chest |
|  Point to the sky | **Messi** | Raise both hands and point upwards |
|  Arms spread wide & low | **Ronaldo** | Spread arms wide at hip level (SIUUU!) |
|  Camera frame | **Son** | Make a camera frame with both hands near face |
|  Open palms | **Diaz** | Hold open palms outward at shoulder height |
|  Hands near ears | **Neymar** | Raise open hands near your ears |

## Requirements

- Python 3.8 – 3.11 (MediaPipe does not support 3.12+ on Windows yet)
- A webcam — built-in, or a phone via DroidCam / Iriun Webcam
- No GPU needed — runs on a mid-range CPU

## Setup & Launching the App

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/football-celebration-detector.git
cd football-celebration-detector

# 2. Create a virtual environment
python -m venv venv
.\venv\Scripts\activate       # Windows PowerShell
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the brand-new elegant World Cup GUI (Recommended)
python main_gui.py

# Alternatively, run the classic OpenCV terminal-based window
python main.py
```

## GUI Features

- ** World Cup Theme:** Designed with a premium burgundy and gold styling, metallic accents, and rounded glassmorphism cards.
- ** Dynamic Camera Selector:** Switch between webcams or enter custom IP camera URLs (for DroidCam) directly in the UI!
- ** Quest Progress Tracker:** Challenge yourself to unlock all 6 player celebrations! A golden progress bar and checklist update dynamically when gestures are recognized.
- ** Real-Time Metrics:** Live tracking displays showing current frame rate (FPS), detected pose landmarks, hand tracking counts, and body visibility status.
- ** Snapshot Flash Effect:** Take screenshots with a glowing flash visual effect.

## Camera Options

In `main.py`, change `CAMERA_SOURCE`:

```python
CAMERA_SOURCE = 0                              # Default webcam
CAMERA_SOURCE = 1                              # Second camera
CAMERA_SOURCE = "http://192.168.x.x:4747/video"  # DroidCam Wi-Fi
```

## Controls

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit the application |
| `S` | Save screenshot to `screenshots/` folder |

## How It Works

```
Webcam Feed
    ↓
OpenCV — captures and mirrors each frame
    ↓
MediaPipe Pose — detects 33 body landmarks
MediaPipe Hands — detects 21 hand landmarks per hand
    ↓
Celebration Detector — classifies body position into a celebration
    ↓
Overlay Renderer — picks the matching player image
    ↓
Output Window — shows camera + player side by side
```

The celebration classifier works by:
1. Checking relative positions of wrists, shoulders, elbows, and nose
2. Calculating distances relative to shoulder width (for scale invariance)
3. Optionally checking finger extension for hand-based celebrations

## Tips for Best Detection

- Make sure your **full upper body** is visible in the frame
- Good lighting helps — avoid backlighting
- Hold celebrations **steady** for ~0.5 seconds
- Works best against a **plain background**
- Stand about **1-2 meters** from the camera

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera won't open | Try changing `CAMERA_SOURCE` to `1` or `2` in `main.py` |
| Celebrations not detecting | Check lighting; make sure full upper body is in frame |
| MediaPipe install fails | Confirm Python 3.8–3.11, not 3.12+ |
| Laggy performance | Close other apps; try reducing `FRAME_WIDTH`/`FRAME_HEIGHT` |

## Tech Stack

- [OpenCV](https://opencv.org/) — Video capture & image processing
- [MediaPipe](https://mediapipe.dev/) — Body pose & hand landmark detection
- [NumPy](https://numpy.org/) — Array manipulation

## License

Free to use, modify, and share.
