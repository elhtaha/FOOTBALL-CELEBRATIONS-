"""
overlay.py
----------
Loads celebration images and composites them beside the camera frame.

Layout (side-by-side)
---------------------
┌─────────────────────┬─────────────────┐
│   Camera feed       │   Player image  │
│   (640 × 480)       │   (320 × 480)   │
└─────────────────────┴─────────────────┘
Total window: 960 × 480

The player panel shows:
  • A dark background with a football theme when no celebration is detected
  • The matching player celebration image when a pose is held
  • A label at the bottom naming the celebration and player
"""

import os
import cv2
import numpy as np

# Maps celebration names to image filenames
CELEBRATION_TO_FILE = {
    "MBAPPE_CROSS":    "mbappe.webp",
    "MESSI_SKYPOINT":  "messi.webp",
    "RONALDO_SIUUU":   "ronaldo.jpeg",
    "SON_CAMERA":      "sonnn.jpeg",
    "DIAZ_OPEN":       "diazz.jpg",
    "NEYMAR_EARS":     "naymar.jpg",
}

# Display labels
CELEBRATION_LABELS = {
    "MBAPPE_CROSS":    "Mbappe - Arms Crossed",
    "MESSI_SKYPOINT":  "Messi - Point to the Sky",
    "RONALDO_SIUUU":   "Ronaldo - SIUUU!",
    "SON_CAMERA":      "Son - Camera Celebration",
    "DIAZ_OPEN":       "Diaz - Open Arms",
    "NEYMAR_EARS":     "Neymar - Playful Ears",
    "NONE":            "Do a celebration!",
}

# Panel styling
PLAYER_PANEL_W = 320
FONT           = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE     = 0.6
FONT_COLOR     = (255, 255, 255)
FONT_THICK     = 1
LABEL_BG       = (20, 20, 20)
ACCENT_COLOR   = (0, 200, 100)    # Green accent for football theme


class OverlayRenderer:
    def __init__(self, images_dir: str = "."):
        self.images: dict[str, np.ndarray | None] = {}
        self._load_images(images_dir)

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_images(self, images_dir: str):
        """Load every celebration image once at startup."""
        for celebration, filename in CELEBRATION_TO_FILE.items():
            path = os.path.join(images_dir, filename)
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                self.images[celebration] = img
                print(f"[Overlay] Loaded: {filename}")
            else:
                self.images[celebration] = None
                print(f"[Overlay] WARNING: {path} not found — will show placeholder")

    def _resize_to_panel(self, img: np.ndarray, panel_h: int) -> np.ndarray:
        """Resize image to fit the panel, keeping aspect ratio."""
        h, w = img.shape[:2]
        scale = min(PLAYER_PANEL_W / w, panel_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _alpha_composite(self, bg: np.ndarray, fg: np.ndarray, y_off: int, x_off: int):
        """Paste fg (BGRA) onto bg (BGR) at (x_off, y_off) using alpha blending."""
        if fg.shape[2] == 4:
            alpha = fg[:, :, 3] / 255.0
            for c in range(3):
                bg[y_off:y_off + fg.shape[0], x_off:x_off + fg.shape[1], c] = (
                    alpha * fg[:, :, c] +
                    (1 - alpha) * bg[y_off:y_off + fg.shape[0], x_off:x_off + fg.shape[1], c]
                )
        else:
            bg[y_off:y_off + fg.shape[0], x_off:x_off + fg.shape[1]] = fg[:, :, :3]

    def _build_player_panel(self, celebration: str, panel_h: int) -> np.ndarray:
        """Return a (panel_h × PLAYER_PANEL_W × 3) BGR panel."""
        # Dark gradient background
        panel = np.zeros((panel_h, PLAYER_PANEL_W, 3), dtype=np.uint8)
        for y in range(panel_h):
            shade = int(15 + 10 * (y / panel_h))
            panel[y, :] = (shade, shade + 3, shade)

        # Add subtle border on left side
        cv2.line(panel, (0, 0), (0, panel_h), ACCENT_COLOR, 2)

        img = self.images.get(celebration)

        if img is not None:
            resized = self._resize_to_panel(img, panel_h - 50)  # leave space for label
            rh, rw = resized.shape[:2]
            y_off = (panel_h - 50 - rh) // 2
            x_off = (PLAYER_PANEL_W - rw) // 2
            self._alpha_composite(panel, resized, max(0, y_off), max(0, x_off))
        elif celebration != "NONE":
            # Placeholder text when image file is missing
            cv2.putText(panel, "Image not found", (10, panel_h // 2),
                        FONT, 0.5, (80, 80, 80), 1)
        else:
            # "Waiting" state — show helpful text with categories
            lines = [
                ("Waiting for a", ACCENT_COLOR),
                ("celebration...", ACCENT_COLOR),
                ("", (150, 150, 150)),
                ("HEAD + HANDS:", (0, 255, 255)),
                ("  Point to sky (Messi)", (150, 150, 150)),
                ("  Camera frame (Son)", (150, 150, 150)),
                ("  Hands at ears (Neymar)", (150, 150, 150)),
                ("", (150, 150, 150)),
                ("FULL BODY:", (100, 200, 255)),
                ("  Cross arms (Mbappe)", (150, 150, 150)),
                ("  Arms wide (Ronaldo)", (150, 150, 150)),
                ("  Open palms (Diaz)", (150, 150, 150)),
            ]
            for i, (line, color) in enumerate(lines):
                cv2.putText(panel, line, (15, 70 + i * 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        # Label bar at the bottom
        label = CELEBRATION_LABELS.get(celebration, "")
        if label:
            bar_y = panel_h - 45
            cv2.rectangle(panel, (0, bar_y), (PLAYER_PANEL_W, panel_h), LABEL_BG, -1)
            # Green accent line above label
            cv2.line(panel, (0, bar_y), (PLAYER_PANEL_W, bar_y), ACCENT_COLOR, 2)
            text_size = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)[0]
            tx = (PLAYER_PANEL_W - text_size[0]) // 2
            cv2.putText(panel, label, (tx, panel_h - 15),
                        FONT, FONT_SCALE, FONT_COLOR, FONT_THICK)

        return panel

    # ── Public ────────────────────────────────────────────────────────────────

    def render(self, camera_frame: np.ndarray, celebration: str) -> np.ndarray:
        """
        Returns the final side-by-side frame:
            [camera_frame | player_panel]
        """
        panel_h = camera_frame.shape[0]
        player_panel = self._build_player_panel(celebration, panel_h)
        return np.hstack([camera_frame, player_panel])
