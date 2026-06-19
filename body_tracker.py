"""
body_tracker.py
---------------
Wraps MediaPipe Tasks (PoseLandmarker + HandLandmarker + FaceLandmarker)
to detect body pose, hand landmarks, and facial features.
Each region is drawn in a distinct color:
  - FACE  → Cyan mesh (478 landmarks: eyes, nose, lips, jawline, eyebrows)
  - BODY  → Green skeleton (33 pose landmarks: shoulders, elbows, hips, etc.)
  - HANDS → Orange/Blue skeleton (21 landmarks per hand: fingers, palm)
Uses the new MediaPipe Tasks API (0.10.20+).
"""
import os
import cv2
import numpy as np
import mediapipe as mp
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
class BodyTracker:
    """Combined Pose + Hands + Face tracker for celebration detection."""
    NOSE            = 0
    LEFT_SHOULDER   = 11
    RIGHT_SHOULDER  = 12
    LEFT_ELBOW      = 13
    RIGHT_ELBOW     = 14
    LEFT_WRIST      = 15
    RIGHT_WRIST     = 16
    LEFT_HIP        = 23
    RIGHT_HIP       = 24
    HAND_WRIST      = 0
    THUMB_TIP       = 4
    THUMB_IP        = 3
    THUMB_MCP       = 2
    INDEX_TIP       = 8
    INDEX_PIP       = 6
    INDEX_MCP       = 5
    MIDDLE_TIP      = 12
    MIDDLE_PIP      = 10
    MIDDLE_MCP      = 9
    RING_TIP        = 16
    RING_PIP        = 14
    RING_MCP        = 13
    PINKY_TIP       = 20
    PINKY_PIP       = 18
    PINKY_MCP       = 17
    FACE_NOSE_TIP       = 1
    FACE_LEFT_EYE       = 33    
    FACE_RIGHT_EYE      = 263   
    FACE_LEFT_EYE_OUT   = 133   
    FACE_RIGHT_EYE_OUT  = 362   
    FACE_MOUTH_LEFT     = 61    
    FACE_MOUTH_RIGHT    = 291   
    FACE_MOUTH_TOP      = 13    
    FACE_MOUTH_BOTTOM   = 14    
    FACE_LEFT_EYEBROW   = 70    
    FACE_RIGHT_EYEBROW  = 300   
    FACE_CHIN            = 152   
    FACE_FOREHEAD        = 10    
    FACE_LEFT_CHEEK      = 234   
    FACE_RIGHT_CHEEK     = 454   
    FACE_LEFT_EAR        = 127   
    FACE_RIGHT_EAR       = 356   
    COLOR_FACE_MESH      = (200, 200, 50)    
    COLOR_FACE_CONTOUR   = (255, 255, 0)     
    COLOR_FACE_EYES      = (0, 255, 255)     
    COLOR_FACE_EYEBROWS  = (0, 200, 200)     
    COLOR_FACE_LIPS      = (100, 100, 255)   
    COLOR_FACE_NOSE      = (200, 255, 200)   
    COLOR_FACE_POINT     = (255, 200, 0)     
    COLOR_BODY_LINE      = (0, 255, 0)       
    COLOR_BODY_POINT     = (0, 0, 255)       
    COLOR_HAND_1         = (255, 100, 50)    
    COLOR_HAND_2         = (50, 100, 255)    
    COLOR_HAND_POINT     = (255, 255, 255)   
    POSE_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  
        (11, 23), (12, 24), (23, 24),                       
        (23, 25), (25, 27), (24, 26), (26, 28),             
        (0, 11), (0, 12),                                    
    ]
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       
        (0, 5), (5, 6), (6, 7), (7, 8),       
        (0, 9), (9, 10), (10, 11), (11, 12),  
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),             
    ]
    FACE_OVAL = [
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
        (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
        (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
        (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
        (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
        (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
        (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
    ]
    FACE_LEFT_EYE_CONTOUR = [
        (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
        (153, 154), (154, 155), (155, 133), (133, 173), (173, 157),
        (157, 158), (158, 159), (159, 160), (160, 161), (161, 246), (246, 33),
    ]
    FACE_RIGHT_EYE_CONTOUR = [
        (263, 249), (249, 390), (390, 373), (373, 374), (374, 380),
        (380, 381), (381, 382), (382, 362), (362, 398), (398, 384),
        (384, 385), (385, 386), (386, 387), (387, 388), (388, 466), (466, 263),
    ]
    FACE_LIPS_OUTER = [
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
        (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
        (291, 409), (409, 270), (270, 269), (269, 267), (267, 0),
        (0, 37), (37, 39), (39, 40), (40, 185), (185, 61),
    ]
    FACE_LIPS_INNER = [
        (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
        (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
        (308, 415), (415, 310), (310, 311), (311, 312), (312, 13),
        (13, 82), (82, 81), (81, 80), (80, 191), (191, 78),
    ]
    FACE_LEFT_EYEBROW = [
        (46, 53), (53, 52), (52, 65), (65, 55), (55, 107),
        (107, 66), (66, 105), (105, 63), (63, 70), (70, 46),
    ]
    FACE_RIGHT_EYEBROW = [
        (276, 283), (283, 282), (282, 295), (295, 285), (285, 336),
        (336, 296), (296, 334), (334, 293), (293, 300), (300, 276),
    ]
    FACE_NOSE = [
        (168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
        (4, 1), (1, 19), (19, 94), (94, 2), (2, 164),
        (164, 0), (0, 11), (11, 12),
    ]
    FACE_LEFT_IRIS = [(468, 469), (469, 470), (470, 471), (471, 468)]
    FACE_RIGHT_IRIS = [(473, 474), (474, 475), (475, 476), (476, 473)]
    def __init__(self, detection_conf=0.5, tracking_conf=0.5,
                 models_dir="models"):
        pose_model_path = os.path.join(models_dir, "pose_landmarker.task")
        hand_model_path = os.path.join(models_dir, "hand_landmarker.task")
        face_model_path = os.path.join(models_dir, "face_landmarker.task")
        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection_conf,
            min_pose_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            pose_options
        )
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            hand_options
        )
        face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=detection_conf,
            min_face_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            face_options
        )
        self._frame_count = 0
    def _draw_face_contour(self, frame, landmarks, connections, color, thickness=1):
        """Draw connected face contour lines."""
        h, w, _ = frame.shape
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                pt1 = (int(landmarks[start_idx].x * w),
                       int(landmarks[start_idx].y * h))
                pt2 = (int(landmarks[end_idx].x * w),
                       int(landmarks[end_idx].y * h))
                cv2.line(frame, pt1, pt2, color, thickness)
    def _draw_face(self, frame, face_landmarks_list):
        """
        Draw detailed facial features on the frame with color-coded regions:
          - Jawline/oval  → Cyan contour
          - Eyes          → Yellow contour
          - Eyebrows      → Dark yellow
          - Lips          → Pink/Red
          - Nose          → Light green
          - Iris          → Bright white
        """
        h, w, _ = frame.shape
        for face_landmarks in face_landmarks_list:
            lm = face_landmarks
            self._draw_face_contour(frame, lm, self.FACE_OVAL,
                                    self.COLOR_FACE_CONTOUR, 2)
            self._draw_face_contour(frame, lm, self.FACE_LEFT_EYE_CONTOUR,
                                    self.COLOR_FACE_EYES, 1)
            self._draw_face_contour(frame, lm, self.FACE_RIGHT_EYE_CONTOUR,
                                    self.COLOR_FACE_EYES, 1)
            self._draw_face_contour(frame, lm, self.FACE_LEFT_EYEBROW,
                                    self.COLOR_FACE_EYEBROWS, 1)
            self._draw_face_contour(frame, lm, self.FACE_RIGHT_EYEBROW,
                                    self.COLOR_FACE_EYEBROWS, 1)
            self._draw_face_contour(frame, lm, self.FACE_LIPS_OUTER,
                                    self.COLOR_FACE_LIPS, 2)
            self._draw_face_contour(frame, lm, self.FACE_LIPS_INNER,
                                    self.COLOR_FACE_LIPS, 1)
            self._draw_face_contour(frame, lm, self.FACE_NOSE,
                                    self.COLOR_FACE_NOSE, 1)
            if len(lm) > 476:
                self._draw_face_contour(frame, lm, self.FACE_LEFT_IRIS,
                                        (255, 255, 255), 1)
                self._draw_face_contour(frame, lm, self.FACE_RIGHT_IRIS,
                                        (255, 255, 255), 1)
                for iris_center in [468, 473]:
                    cx = int(lm[iris_center].x * w)
                    cy = int(lm[iris_center].y * h)
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
            key_points = [
                self.FACE_NOSE_TIP, self.FACE_LEFT_EYE, self.FACE_RIGHT_EYE,
                self.FACE_LEFT_EYE_OUT, self.FACE_RIGHT_EYE_OUT,
                self.FACE_MOUTH_LEFT, self.FACE_MOUTH_RIGHT,
                self.FACE_MOUTH_TOP, self.FACE_MOUTH_BOTTOM,
                self.FACE_CHIN, self.FACE_FOREHEAD,
                self.FACE_LEFT_EAR, self.FACE_RIGHT_EAR,
            ]
            for idx in key_points:
                if idx < len(lm):
                    cx = int(lm[idx].x * w)
                    cy = int(lm[idx].y * h)
                    cv2.circle(frame, (cx, cy), 3, self.COLOR_FACE_POINT, -1)
            if self.FACE_FOREHEAD < len(lm):
                fx = int(lm[self.FACE_FOREHEAD].x * w)
                fy = int(lm[self.FACE_FOREHEAD].y * h) - 15
                cv2.putText(frame, "FACE", (fx - 20, fy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            self.COLOR_FACE_CONTOUR, 1)
    def _draw_pose(self, frame, pose_landmarks_list):
        """Draw pose skeleton on the frame (green = body)."""
        h, w, _ = frame.shape
        for pose_landmarks in pose_landmarks_list:
            landmarks = pose_landmarks
            for start_idx, end_idx in self.POSE_CONNECTIONS:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    pt1 = (int(landmarks[start_idx].x * w),
                           int(landmarks[start_idx].y * h))
                    pt2 = (int(landmarks[end_idx].x * w),
                           int(landmarks[end_idx].y * h))
                    cv2.line(frame, pt1, pt2, self.COLOR_BODY_LINE, 2)
            for i, lm in enumerate(landmarks):
                if i > 10:  
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, self.COLOR_BODY_POINT, -1)
            if self.LEFT_SHOULDER < len(landmarks):
                bx = int(landmarks[self.LEFT_SHOULDER].x * w) + 10
                by = int(landmarks[self.LEFT_SHOULDER].y * h) - 10
                cv2.putText(frame, "BODY", (bx, by),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            self.COLOR_BODY_LINE, 1)
    def _draw_hands(self, frame, hand_landmarks_list):
        """Draw hand skeleton on the frame (orange/blue = hands)."""
        h, w, _ = frame.shape
        colors = [self.COLOR_HAND_1, self.COLOR_HAND_2]
        hand_labels = ["L-HAND", "R-HAND"]
        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            color = colors[idx % len(colors)]
            landmarks = hand_landmarks
            for start_idx, end_idx in self.HAND_CONNECTIONS:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    pt1 = (int(landmarks[start_idx].x * w),
                           int(landmarks[start_idx].y * h))
                    pt2 = (int(landmarks[end_idx].x * w),
                           int(landmarks[end_idx].y * h))
                    cv2.line(frame, pt1, pt2, color, 2)
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, self.COLOR_HAND_POINT, -1)
            if len(landmarks) > 0:
                hx = int(landmarks[0].x * w)
                hy = int(landmarks[0].y * h) + 20
                cv2.putText(frame, hand_labels[idx % 2], (hx - 20, hy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    def process(self, frame_bgr):
        """
        Run pose + hand + face detection on a BGR frame.
        Returns
        -------
        pose_landmarks : list of (x, y, z, visibility) tuples in normalised coords [0,1],
                         or None if no body found.
        hand_landmarks_list : list of hand landmark lists, each with 21 (x, y) tuples
                              in pixel coords. Empty list if no hands found.
        face_landmarks : list of (x, y, z) tuples for 478 facial landmarks in normalised
                         coords, or None if no face found.
        annotated_frame : frame with face mesh + body skeleton + hand skeleton drawn,
                          each in a distinct color.
        """
        self._frame_count += 1
        timestamp_ms = self._frame_count
        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        annotated = frame_bgr.copy()
        face_lm = None
        try:
            face_result = self.face_landmarker.detect_for_video(
                mp_image, timestamp_ms
            )
            if face_result.face_landmarks:
                self._draw_face(annotated, face_result.face_landmarks)
                raw_face = face_result.face_landmarks[0]
                face_lm = [
                    (lm.x, lm.y, lm.z)
                    for lm in raw_face
                ]
        except Exception:
            pass
        pose_lm = None
        try:
            pose_result = self.pose_landmarker.detect_for_video(
                mp_image, timestamp_ms
            )
            if pose_result.pose_landmarks:
                self._draw_pose(annotated, pose_result.pose_landmarks)
                raw_lm = pose_result.pose_landmarks[0]
                pose_lm = [
                    (lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0)
                    for lm in raw_lm
                ]
        except Exception:
            pass
        hand_lm_list = []
        try:
            hand_result = self.hand_landmarker.detect_for_video(
                mp_image, timestamp_ms
            )
            if hand_result.hand_landmarks:
                self._draw_hands(annotated, hand_result.hand_landmarks)
                for hand_landmarks in hand_result.hand_landmarks:
                    landmarks = [
                        (lm.x * w, lm.y * h)
                        for lm in hand_landmarks
                    ]
                    hand_lm_list.append(landmarks)
        except Exception:
            pass
        legend_x = w - 160
        legend_y = 45
        cv2.rectangle(annotated, (legend_x - 5, legend_y - 15),
                      (w - 5, legend_y + 55), (0, 0, 0), -1)
        cv2.rectangle(annotated, (legend_x - 5, legend_y - 15),
                      (w - 5, legend_y + 55), (60, 60, 60), 1)
        cv2.circle(annotated, (legend_x + 5, legend_y), 5,
                   self.COLOR_FACE_CONTOUR, -1)
        cv2.putText(annotated, "Face", (legend_x + 15, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_FACE_CONTOUR, 1)
        cv2.circle(annotated, (legend_x + 5, legend_y + 20), 5,
                   self.COLOR_BODY_LINE, -1)
        cv2.putText(annotated, "Body", (legend_x + 15, legend_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_BODY_LINE, 1)
        cv2.circle(annotated, (legend_x + 5, legend_y + 40), 5,
                   self.COLOR_HAND_1, -1)
        cv2.putText(annotated, "Hands", (legend_x + 15, legend_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_HAND_1, 1)
        return pose_lm, hand_lm_list, face_lm, annotated
    def close(self):
        self.pose_landmarker.close()
        self.hand_landmarker.close()
        self.face_landmarker.close()