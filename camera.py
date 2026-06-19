import cv2


class Camera:
    def __init__(self, source=0, width=640, height=480):
        """
            0  → default webcam / DroidCam USB
            "http://192.168.x.x:4747/video" → DroidCam over Wi-Fi
        """
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[Camera] Cannot open source '{source}'.\n"
                "  • If using DroidCam USB  → try source=1 or source=2\n"
                "  • If using DroidCam Wi-Fi → pass the full URL as source\n"
                "  • If using Iriun Webcam   → make sure the app is running first"
            )

    def read(self):
        """Return a mirrored BGR frame, or None on failure."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)   # mirror → selfie mode

    def release(self):
        self.cap.release()
