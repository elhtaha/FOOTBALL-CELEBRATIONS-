 
import cv2
import time
import os
from camera               import Camera
from body_tracker          import BodyTracker
from celebration_detector  import CelebrationDetector
from overlay               import OverlayRenderer
CAMERA_SOURCE     = 0          
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
COOLDOWN_SECONDS  = 1.5        
WINDOW_TITLE      = "Football Celebration Detector  |  Q = quit  |  S = screenshot"
SCREENSHOT_DIR    = "screenshots"
def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    camera   = Camera(CAMERA_SOURCE, FRAME_WIDTH, FRAME_HEIGHT)
    tracker  = BodyTracker(detection_conf=0.7, tracking_conf=0.7)
    detector = CelebrationDetector()
    renderer = OverlayRenderer(images_dir=".")
    displayed_celebration = "NONE"
    last_trigger_time     = 0.0
    print("=" * 60)
    print("  FOOTBALL CELEBRATION DETECTOR")
    print("=" * 60)
    print()
    print("  Celebrations to try:")
    print("  -----------------------------------------")
    print("  [1] Cross arms on chest      -> Mbappe")
    print("  [2] Point both hands to sky  -> Messi")
    print("  [3] Spread arms wide & low   -> Ronaldo")
    print("  [4] Camera frame near face   -> Son")
    print("  [5] Open palms outward       -> Diaz")
    print("  [6] Hands near ears (open)   -> Neymar")
    print()
    print("  Press Q or ESC to quit, S to screenshot.")
    print("=" * 60)
    while True:
        frame = camera.read()
        if frame is None:
            print("[Main] Frame read failed — check your camera source.")
            break
        pose_lm, hand_lm_list, face_lm, annotated_frame = tracker.process(frame)
        raw_celebration = detector.detect(pose_lm, hand_lm_list)
        now = time.time()
        if raw_celebration != "NONE":
            displayed_celebration = raw_celebration
            last_trigger_time = now
        elif now - last_trigger_time > COOLDOWN_SECONDS:
            displayed_celebration = "NONE"
        cv2.rectangle(annotated_frame, (0, 0), (FRAME_WIDTH, 35), (0, 0, 0), -1)
        cv2.putText(annotated_frame, "Football Celebration Detector",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 100), 2)
        fps_text = f"FPS: {int(1 / max(time.time() - now, 1e-6))}"
        cv2.putText(annotated_frame, fps_text, (FRAME_WIDTH - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if displayed_celebration != "NONE":
            cv2.rectangle(annotated_frame, (0, FRAME_HEIGHT - 40),
                         (FRAME_WIDTH, FRAME_HEIGHT), (0, 150, 80), -1)
            cv2.putText(annotated_frame,
                       f"Detected: {displayed_celebration.replace('_', ' ')}",
                       (10, FRAME_HEIGHT - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        output = renderer.render(annotated_frame, displayed_celebration)
        cv2.imshow(WINDOW_TITLE, output)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   
            break
        if key in (ord('s'), ord('S')):        
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"celebration_{ts}.png")
            cv2.imwrite(path, output)
            print(f"[Main] Screenshot saved → {path}")
    camera.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("[Main] Closed.")
if __name__ == "__main__":
    main()