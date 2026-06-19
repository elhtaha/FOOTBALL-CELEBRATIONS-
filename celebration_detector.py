
import math
from body_tracker import BodyTracker as BT
MIN_VISIBILITY = 0.5
def _distance(p1, p2):
    """Euclidean distance between two (x, y) points (normalised coords)."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
def _finger_extended(hand_lm, tip_idx, pip_idx):
    """Check if a finger is extended (distance from wrist to tip > wrist to PIP)."""
    wrist = hand_lm[BT.HAND_WRIST]
    dist_tip = _distance(wrist, hand_lm[tip_idx])
    dist_pip = _distance(wrist, hand_lm[pip_idx])
    return dist_tip > dist_pip * 1.1
def _count_extended_fingers(hand_lm):
    """Count how many fingers are extended (excluding thumb)."""
    count = 0
    pairs = [
        (BT.INDEX_TIP, BT.INDEX_PIP),
        (BT.MIDDLE_TIP, BT.MIDDLE_PIP),
        (BT.RING_TIP, BT.RING_PIP),
        (BT.PINKY_TIP, BT.PINKY_PIP),
    ]
    for tip, pip in pairs:
        if _finger_extended(hand_lm, tip, pip):
            count += 1
    return count
def _thumb_extended(hand_lm):
    """Check if thumb is extended (tip further from wrist than IP joint)."""
    thumb_tip = hand_lm[BT.THUMB_TIP]
    thumb_ip = hand_lm[BT.THUMB_IP]
    wrist = hand_lm[BT.HAND_WRIST]
    return abs(thumb_tip[0] - wrist[0]) > abs(thumb_ip[0] - wrist[0])
def _is_full_body_visible(pose_lm):
    """
    Check if the whole body is visible in the camera frame.
    Returns True only if hips have good visibility scores and are in frame.
    """
    if pose_lm is None or len(pose_lm) < 25:
        return False
    left_hip  = pose_lm[BT.LEFT_HIP]
    right_hip = pose_lm[BT.RIGHT_HIP]
    left_hip_visible  = left_hip[3] > MIN_VISIBILITY
    right_hip_visible = right_hip[3] > MIN_VISIBILITY
    left_hip_in_frame  = 0.05 < left_hip[1] < 0.95
    right_hip_in_frame = 0.05 < right_hip[1] < 0.95
    return (left_hip_visible and right_hip_visible and
            left_hip_in_frame and right_hip_in_frame)
class CelebrationDetector:
    """Classifies body poses and hand gestures into football celebrations."""
    UPPER_BODY_CELEBRATIONS = ("MESSI_SKYPOINT", "SON_CAMERA", "NEYMAR_EARS")
    FULL_BODY_CELEBRATIONS = ("MBAPPE_CROSS", "RONALDO_SIUUU", "DIAZ_OPEN")
    CELEBRATIONS = UPPER_BODY_CELEBRATIONS + FULL_BODY_CELEBRATIONS + ("NONE",)
    def detect(self, pose_lm, hand_lm_list):
        """
        Parameters
        ----------
        pose_lm : list of (x, y, z, visibility) tuples in normalised coords [0,1],
                   or None if no body detected.
        hand_lm_list : list of hand landmark lists, each with 21 (x, y) tuples
                       in pixel coords. Can be empty.
        Returns
        -------
        str : One of the strings in CELEBRATIONS.
        """
        if pose_lm is None:
            return "NONE"
        left_shoulder  = pose_lm[BT.LEFT_SHOULDER]
        right_shoulder = pose_lm[BT.RIGHT_SHOULDER]
        left_elbow     = pose_lm[BT.LEFT_ELBOW]
        right_elbow    = pose_lm[BT.RIGHT_ELBOW]
        left_wrist     = pose_lm[BT.LEFT_WRIST]
        right_wrist    = pose_lm[BT.RIGHT_WRIST]
        nose           = pose_lm[BT.NOSE]
        shoulder_width = _distance(left_shoulder, right_shoulder)
        if shoulder_width < 0.01:
            return "NONE"
        chest_x = (left_shoulder[0] + right_shoulder[0]) / 2
        chest_y = (left_shoulder[1] + right_shoulder[1]) / 2
        full_body_visible = _is_full_body_visible(pose_lm)
        both_wrists_above_head = (
            left_wrist[1] < nose[1] - shoulder_width * 0.3 and
            right_wrist[1] < nose[1] - shoulder_width * 0.3
        )
        wrists_spread_up = abs(left_wrist[0] - right_wrist[0]) > shoulder_width * 0.5
        if both_wrists_above_head and wrists_spread_up:
            if len(hand_lm_list) >= 1:
                pointing_up = False
                for hand_lm in hand_lm_list:
                    if 1 <= _count_extended_fingers(hand_lm) <= 2:
                        pointing_up = True
                        break
                if pointing_up:
                    return "MESSI_SKYPOINT"
        one_high_one_low = (
            (left_wrist[1] < nose[1] and right_wrist[1] > nose[1]) or
            (right_wrist[1] < nose[1] and left_wrist[1] > nose[1])
        )
        opposite_sides_horiz = (
            (left_wrist[0] - nose[0]) * (right_wrist[0] - nose[0]) < 0
        )
        both_wrists_at_face = (
            abs(left_wrist[1] - nose[1]) < shoulder_width * 0.9 and
            abs(right_wrist[1] - nose[1]) < shoulder_width * 0.9
        )
        wrists_above_shoulders = (
            left_wrist[1] < left_shoulder[1] and
            right_wrist[1] < right_shoulder[1]
        )
        wrists_visible = (
            left_wrist[3] > 0.15 and
            right_wrist[3] > 0.15
        )
        both_hands_detected = len(hand_lm_list) >= 2
        both_near_nose_horiz = (
            abs(left_wrist[0] - nose[0]) < shoulder_width * 0.9 and
            abs(right_wrist[0] - nose[0]) < shoulder_width * 0.9
        )
        if (both_wrists_at_face and wrists_above_shoulders and wrists_visible and 
                both_hands_detected and both_near_nose_horiz and one_high_one_low and opposite_sides_horiz):
            has_camera_shape = False
            for hand_lm in hand_lm_list:
                if _count_extended_fingers(hand_lm) <= 3:
                    has_camera_shape = True
                    break
            if has_camera_shape:
                return "SON_CAMERA"
        wrists_at_head_height = (
            left_wrist[1] < left_shoulder[1] and
            right_wrist[1] < right_shoulder[1] and
            left_wrist[1] < nose[1] + shoulder_width * 0.2 and
            right_wrist[1] < nose[1] + shoulder_width * 0.2
        )
        wrists_on_opposite_sides = (
            (left_wrist[0] - nose[0]) * (right_wrist[0] - nose[0]) < 0
        )
        wrists_spread_at_ears = abs(left_wrist[0] - right_wrist[0]) > shoulder_width * 0.8
        wrists_far_from_nose = (
            abs(left_wrist[0] - nose[0]) > shoulder_width * 0.35 and
            abs(right_wrist[0] - nose[0]) > shoulder_width * 0.35
        )
        wrists_above_elbows = (
            left_wrist[1] < left_elbow[1] and
            right_wrist[1] < right_elbow[1]
        )
        wrists_close_to_ears = abs(left_wrist[0] - right_wrist[0]) < shoulder_width * 1.8
        if (wrists_at_head_height and wrists_on_opposite_sides and
                wrists_spread_at_ears and wrists_far_from_nose and wrists_above_elbows and wrists_close_to_ears):
            has_open_hand = False
            if len(hand_lm_list) >= 1:
                for hand_lm in hand_lm_list:
                    if _count_extended_fingers(hand_lm) >= 4:
                        has_open_hand = True
                        break
            if has_open_hand:
                return "NEYMAR_EARS"
        upper_body_visible = (
            left_elbow[3] > 0.3 and right_elbow[3] > 0.3 and
            left_wrist[3] > 0.3 and right_wrist[3] > 0.3 and
            0.02 < left_elbow[1] < 0.98 and 0.02 < right_elbow[1] < 0.98
        )
        if not upper_body_visible:
            return "NONE"
        wrists_crossed = (
            abs(left_wrist[0] - right_shoulder[0]) < shoulder_width * 0.6 and
            abs(right_wrist[0] - left_shoulder[0]) < shoulder_width * 0.6
        )
        wrists_near_chest = (
            abs(left_wrist[1] - chest_y) < shoulder_width * 0.8 and
            abs(right_wrist[1] - chest_y) < shoulder_width * 0.8
        )
        if wrists_crossed and wrists_near_chest:
            return "MBAPPE_CROSS"
        left_hip = pose_lm[BT.LEFT_HIP]
        right_hip = pose_lm[BT.RIGHT_HIP]
        wrists_near_hips = (
            left_wrist[1] > left_hip[1] - shoulder_width * 0.5 and
            right_wrist[1] > right_hip[1] - shoulder_width * 0.5
        )
        wrists_very_wide = (
            abs(left_wrist[0] - right_wrist[0]) > shoulder_width * 1.8
        )
        if full_body_visible and wrists_near_hips and wrists_very_wide:
            return "RONALDO_SIUUU"
        left_elbow = pose_lm[BT.LEFT_ELBOW]
        right_elbow = pose_lm[BT.RIGHT_ELBOW]
        wrists_above_elbows = (
            left_wrist[1] < left_elbow[1] and
            right_wrist[1] < right_elbow[1]
        )
        elbows_near_body = (
            abs(left_elbow[0] - left_shoulder[0]) < shoulder_width * 0.8 and
            abs(right_elbow[0] - right_shoulder[0]) < shoulder_width * 0.8
        )
        wrists_out_to_sides = (
            abs(left_wrist[0] - right_wrist[0]) > abs(left_elbow[0] - right_elbow[0]) + shoulder_width * 0.2
        )
        wrists_near_shoulders_y = (
            abs(left_wrist[1] - left_shoulder[1]) < shoulder_width * 0.8 and
            abs(right_wrist[1] - right_shoulder[1]) < shoulder_width * 0.8
        )
        if wrists_above_elbows and elbows_near_body and wrists_out_to_sides and wrists_near_shoulders_y:
            has_open_hand_diaz = False
            for hand_lm in hand_lm_list:
                if _count_extended_fingers(hand_lm) >= 3:
                    has_open_hand_diaz = True
                    break
            if has_open_hand_diaz:
                return "DIAZ_OPEN"
        return "NONE"