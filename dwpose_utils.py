"""
DWPose utilities — skeleton extraction, pose-to-prompt, and pose validation.

Used for body consistency: extract pose from reference photo, enrich prompt
with body position description, and validate output pose matches input.

Keypoint format: COCO 18-point body + 2x21 hands + 68 face landmarks.
We only use the 18 body keypoints for pose description and validation.
"""

import math
import numpy as np

# COCO 18-point body keypoint indices
KEYPOINT_NAMES = {
    0: "nose",
    1: "neck",
    2: "right_shoulder",
    3: "right_elbow",
    4: "right_wrist",
    5: "left_shoulder",
    6: "left_elbow",
    7: "left_wrist",
    8: "right_hip",
    9: "right_knee",
    10: "right_ankle",
    11: "left_hip",
    12: "left_knee",
    13: "left_ankle",
    14: "right_eye",
    15: "left_eye",
    16: "right_ear",
    17: "left_ear",
}

_dwpose_detector = None


def load_dwpose(device="cuda"):
    """Load DWPose detector (cached)."""
    global _dwpose_detector
    if _dwpose_detector is not None:
        return

    import os
    print("[DWPose] Loading DWPose detector...", flush=True)

    from easy_dwpose import DWposeDetector
    _dwpose_detector = DWposeDetector(device=device)

    print("[DWPose] Detector loaded", flush=True)


def extract_skeleton(image, device="cuda"):
    """
    Extract body skeleton keypoints from a PIL Image.

    Args:
        image: PIL.Image.Image
        device: torch device

    Returns:
        dict with:
          - keypoints: np.array shape (18, 3) — x, y, confidence per body keypoint
          - pose_image: PIL.Image — rendered skeleton visualization
          - raw_result: full DWPose output (body + hands + face)
        Or None if no person detected.
    """
    load_dwpose(device)

    # DWPose returns pose image and optionally keypoints
    # Using the detector's __call__ which returns PIL image
    pose_image = _dwpose_detector(
        image,
        output_type="pil",
        include_hands=True,
        include_face=False,  # face landmarks not needed for body pose
    )

    # Also get raw keypoints for analysis
    import torch
    img_np = np.array(image)

    # easy_dwpose internal: get keypoints directly
    try:
        result = _dwpose_detector.detect_poses(image)
        if result is None or len(result) == 0:
            print("[DWPose] No person detected", flush=True)
            return None

        # Pick the largest/most confident person
        best = result[0]
        if hasattr(best, 'body'):
            body_kps = np.array(best.body.keypoints)  # (18, 3) x,y,conf
        elif isinstance(best, dict) and 'body' in best:
            body_kps = np.array(best['body'])
        else:
            # Fallback: try to extract from raw result
            body_kps = np.array(best)[:18]

    except (AttributeError, TypeError):
        # easy_dwpose API may vary — fallback to image-only mode
        print("[DWPose] Could not extract raw keypoints, using image only", flush=True)
        return {
            "keypoints": None,
            "pose_image": pose_image,
            "raw_result": None,
        }

    # Normalize keypoints to 0-1 range (relative to image dimensions)
    h, w = img_np.shape[:2]
    if body_kps is not None and len(body_kps) >= 18:
        norm_kps = body_kps.copy()
        norm_kps[:, 0] /= w  # x
        norm_kps[:, 1] /= h  # y
    else:
        norm_kps = None

    print(f"[DWPose] Extracted {len(body_kps) if body_kps is not None else 0} body keypoints", flush=True)

    return {
        "keypoints": norm_kps,  # normalized (18, 3)
        "keypoints_abs": body_kps,  # absolute pixel coords
        "pose_image": pose_image,
        "image_size": (w, h),
    }


# ─── Pose-to-Prompt ───
# Translate skeleton keypoints into natural language body position descriptions.

def _angle(a, b, c):
    """Calculate angle at point b given three 2D points."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))


def _is_visible(kp, min_conf=0.3):
    """Check if a keypoint is visible (confidence > threshold)."""
    return len(kp) >= 3 and kp[2] > min_conf


def keypoints_to_pose_description(keypoints):
    """
    Convert normalized body keypoints (18, 3) to a natural language pose description.

    Returns a string like:
    "standing facing camera, left hand on hip, right arm relaxed at side,
     weight on right leg, head tilted slightly left"
    """
    if keypoints is None or len(keypoints) < 18:
        return ""

    kps = keypoints
    parts = []

    # ─── Body orientation ───
    l_shoulder = kps[5]
    r_shoulder = kps[2]
    if _is_visible(l_shoulder) and _is_visible(r_shoulder):
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        if shoulder_width < 0.08:
            parts.append("facing sideways")
        elif shoulder_width > 0.2:
            parts.append("facing camera")
        else:
            # Determine which shoulder is closer (larger x-spread on one side)
            mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
            nose = kps[0]
            if _is_visible(nose):
                if nose[0] < mid_x - 0.03:
                    parts.append("turned slightly right")
                elif nose[0] > mid_x + 0.03:
                    parts.append("turned slightly left")
                else:
                    parts.append("facing slightly angled")
            else:
                parts.append("three-quarter view")

    # ─── Standing / Sitting / Lying ───
    l_hip = kps[11]
    r_hip = kps[8]
    l_knee = kps[12]
    r_knee = kps[9]
    l_ankle = kps[13]
    r_ankle = kps[10]

    hips_visible = _is_visible(l_hip) and _is_visible(r_hip)
    knees_visible = _is_visible(l_knee) and _is_visible(r_knee)
    ankles_visible = _is_visible(l_ankle) and _is_visible(r_ankle)

    if hips_visible and knees_visible:
        hip_y = (l_hip[1] + r_hip[1]) / 2
        knee_y = (l_knee[1] + r_knee[1]) / 2
        hip_knee_dist = knee_y - hip_y

        if hip_knee_dist < 0.05:
            # Hips and knees at similar height → sitting or lying
            if hip_y > 0.6:
                parts.append("lying down")
            else:
                parts.append("seated")
        else:
            # Check knee bend
            if ankles_visible:
                l_angle = _angle(l_hip[:2], l_knee[:2], l_ankle[:2]) if _is_visible(l_knee) and _is_visible(l_ankle) else 180
                r_angle = _angle(r_hip[:2], r_knee[:2], r_ankle[:2]) if _is_visible(r_knee) and _is_visible(r_ankle) else 180

                if l_angle < 140 or r_angle < 140:
                    parts.append("standing with bent knee")
                else:
                    parts.append("standing")
            else:
                parts.append("standing")
    elif hips_visible:
        parts.append("upper body visible")

    # ─── Arms ───
    for side, shoulder_idx, elbow_idx, wrist_idx, hip_idx in [
        ("left", 5, 6, 7, 11),
        ("right", 2, 3, 4, 8),
    ]:
        shoulder = kps[shoulder_idx]
        elbow = kps[elbow_idx]
        wrist = kps[wrist_idx]
        hip = kps[hip_idx]

        if not _is_visible(shoulder):
            continue

        if _is_visible(elbow) and _is_visible(wrist):
            arm_angle = _angle(shoulder[:2], elbow[:2], wrist[:2])

            # Check if hand is near hip (hand on hip)
            if _is_visible(hip):
                wrist_to_hip = math.sqrt((wrist[0] - hip[0])**2 + (wrist[1] - hip[1])**2)
                if wrist_to_hip < 0.08:
                    parts.append(f"{side} hand on hip")
                    continue

            # Check if arm is raised
            if wrist[1] < shoulder[1] - 0.1:
                if wrist[1] < kps[0][1]:  # above head
                    parts.append(f"{side} arm raised above head")
                else:
                    parts.append(f"{side} arm raised")
            elif arm_angle < 100:
                parts.append(f"{side} arm bent")
            elif abs(wrist[1] - hip[1]) < 0.1 if _is_visible(hip) else False:
                parts.append(f"{side} arm at side")

        elif _is_visible(elbow):
            if elbow[1] < shoulder[1]:
                parts.append(f"{side} arm raised")

    # ─── Head tilt ───
    nose = kps[0]
    neck = kps[1]
    if _is_visible(nose) and _is_visible(neck):
        head_tilt = nose[0] - neck[0]
        if abs(head_tilt) > 0.03:
            direction = "left" if head_tilt < 0 else "right"
            parts.append(f"head tilted slightly {direction}")

    # ─── Looking direction ───
    l_eye = kps[15]
    r_eye = kps[14]
    if _is_visible(l_eye) and _is_visible(r_eye) and _is_visible(nose):
        eye_center = ((l_eye[0] + r_eye[0]) / 2, (l_eye[1] + r_eye[1]) / 2)
        if _is_visible(l_shoulder) and _is_visible(r_shoulder):
            body_center = (l_shoulder[0] + r_shoulder[0]) / 2
            gaze_offset = eye_center[0] - body_center
            if gaze_offset < -0.05:
                parts.append("looking right")
            elif gaze_offset > 0.05:
                parts.append("looking left")
            else:
                parts.append("looking at camera")

    # ─── Weight distribution (which leg bears weight) ───
    if ankles_visible and hips_visible:
        hip_center_x = (l_hip[0] + r_hip[0]) / 2
        l_ankle_offset = abs(l_ankle[0] - hip_center_x)
        r_ankle_offset = abs(r_ankle[0] - hip_center_x)
        if l_ankle_offset < r_ankle_offset - 0.04:
            parts.append("weight on left leg")
        elif r_ankle_offset < l_ankle_offset - 0.04:
            parts.append("weight on right leg")

    if not parts:
        return ""

    return ", ".join(parts)


# ─── Pose Validation ───
# Compare two skeleton extractions to check if the generated image matches
# the reference pose.

def compute_pose_similarity(ref_keypoints, gen_keypoints, min_conf=0.3):
    """
    Compare two sets of normalized body keypoints.

    Args:
        ref_keypoints: np.array (18, 3) — reference pose keypoints (normalized)
        gen_keypoints: np.array (18, 3) — generated image keypoints (normalized)
        min_conf: minimum confidence to consider a keypoint valid

    Returns:
        dict with:
          - score: float 0.0-1.0 (1.0 = perfect match)
          - matched_points: int — number of keypoints compared
          - total_points: int — total possible keypoints
          - per_joint: dict of per-joint distances
          - verdict: "pass" | "warn" | "fail"
    """
    if ref_keypoints is None or gen_keypoints is None:
        return {
            "score": None,
            "matched_points": 0,
            "total_points": 18,
            "verdict": "skip",
            "reason": "keypoints not available",
        }

    if len(ref_keypoints) < 18 or len(gen_keypoints) < 18:
        return {
            "score": None,
            "matched_points": 0,
            "total_points": 18,
            "verdict": "skip",
            "reason": "insufficient keypoints",
        }

    distances = []
    per_joint = {}
    matched = 0

    for i in range(18):
        ref_kp = ref_keypoints[i]
        gen_kp = gen_keypoints[i]

        if not _is_visible(ref_kp, min_conf) or not _is_visible(gen_kp, min_conf):
            continue

        # Euclidean distance in normalized coords (0-1)
        dist = math.sqrt((ref_kp[0] - gen_kp[0])**2 + (ref_kp[1] - gen_kp[1])**2)
        distances.append(dist)
        per_joint[KEYPOINT_NAMES[i]] = round(dist, 4)
        matched += 1

    if matched == 0:
        return {
            "score": None,
            "matched_points": 0,
            "total_points": 18,
            "verdict": "skip",
            "reason": "no overlapping visible keypoints",
        }

    # Score: convert average distance to 0-1 similarity
    # Distance 0 = perfect match (score 1.0)
    # Distance 0.15 = ~50% match
    # Distance 0.3+ = poor match (score ~0)
    avg_dist = np.mean(distances)
    max_dist = np.max(distances)

    # Sigmoid-like scoring: exponential decay
    score = math.exp(-avg_dist * 10)  # dist=0 → 1.0, dist=0.07 → 0.5, dist=0.15 → 0.22

    # Verdict thresholds
    if score >= 0.6 and max_dist < 0.15:
        verdict = "pass"
    elif score >= 0.35:
        verdict = "warn"
    else:
        verdict = "fail"

    return {
        "score": round(score, 3),
        "avg_distance": round(avg_dist, 4),
        "max_distance": round(max_dist, 4),
        "matched_points": matched,
        "total_points": 18,
        "per_joint": per_joint,
        "verdict": verdict,
    }


def compute_proportion_similarity(ref_keypoints, gen_keypoints, min_conf=0.3):
    """
    Compare body PROPORTIONS between reference and generated image.
    This catches body consistency issues even when pose differs:
    e.g., torso too long, legs too short, shoulders too wide.

    Returns proportion similarity score (0-1).
    """
    if ref_keypoints is None or gen_keypoints is None:
        return {"score": None, "verdict": "skip"}

    # Key proportion ratios to compare
    proportion_pairs = [
        # (name, point_a, point_b) — measure distance between these points
        ("shoulder_width", 2, 5),       # right shoulder ↔ left shoulder
        ("torso_length", 1, 8),         # neck ↔ right hip
        ("right_upper_arm", 2, 3),      # shoulder ↔ elbow
        ("right_forearm", 3, 4),        # elbow ↔ wrist
        ("left_upper_arm", 5, 6),
        ("left_forearm", 6, 7),
        ("right_thigh", 8, 9),          # hip ↔ knee
        ("right_shin", 9, 10),          # knee ↔ ankle
        ("left_thigh", 11, 12),
        ("left_shin", 12, 13),
        ("hip_width", 8, 11),           # right hip ↔ left hip
    ]

    ref_lengths = {}
    gen_lengths = {}

    for name, idx_a, idx_b in proportion_pairs:
        ref_a, ref_b = ref_keypoints[idx_a], ref_keypoints[idx_b]
        gen_a, gen_b = gen_keypoints[idx_a], gen_keypoints[idx_b]

        if (_is_visible(ref_a, min_conf) and _is_visible(ref_b, min_conf) and
            _is_visible(gen_a, min_conf) and _is_visible(gen_b, min_conf)):
            ref_lengths[name] = math.sqrt((ref_a[0]-ref_b[0])**2 + (ref_a[1]-ref_b[1])**2)
            gen_lengths[name] = math.sqrt((gen_a[0]-gen_b[0])**2 + (gen_a[1]-gen_b[1])**2)

    if len(ref_lengths) < 3:
        return {"score": None, "verdict": "skip", "reason": "insufficient proportions"}

    # Normalize all lengths by torso length (scale-invariant comparison)
    ref_norm = ref_lengths.get("torso_length", 1.0)
    gen_norm = gen_lengths.get("torso_length", 1.0)

    if ref_norm < 0.01 or gen_norm < 0.01:
        return {"score": None, "verdict": "skip", "reason": "torso not visible"}

    ratios = {}
    diffs = []
    for name in ref_lengths:
        if name == "torso_length":
            continue
        ref_ratio = ref_lengths[name] / ref_norm
        gen_ratio = gen_lengths[name] / gen_norm
        diff = abs(ref_ratio - gen_ratio) / (ref_ratio + 1e-8)
        ratios[name] = {"ref": round(ref_ratio, 3), "gen": round(gen_ratio, 3), "diff": round(diff, 3)}
        diffs.append(diff)

    if not diffs:
        return {"score": None, "verdict": "skip"}

    avg_diff = np.mean(diffs)
    score = max(0, 1.0 - avg_diff * 2)  # 0% diff → 1.0, 50% diff → 0.0

    if score >= 0.7:
        verdict = "pass"
    elif score >= 0.5:
        verdict = "warn"
    else:
        verdict = "fail"

    return {
        "score": round(score, 3),
        "proportions": ratios,
        "verdict": verdict,
    }


def validate_pose(ref_image, gen_image, device="cuda"):
    """
    Full pose validation: extract skeletons from both images and compare.

    Args:
        ref_image: PIL.Image — reference/input image
        gen_image: PIL.Image — generated output image
        device: torch device

    Returns:
        dict with pose_match, proportion_match, and overall verdict
    """
    import time
    t0 = time.time()

    ref_skeleton = extract_skeleton(ref_image, device=device)
    gen_skeleton = extract_skeleton(gen_image, device=device)

    ref_kps = ref_skeleton["keypoints"] if ref_skeleton else None
    gen_kps = gen_skeleton["keypoints"] if gen_skeleton else None

    pose_match = compute_pose_similarity(ref_kps, gen_kps)
    proportion_match = compute_proportion_similarity(ref_kps, gen_kps)

    # Overall verdict
    verdicts = [pose_match.get("verdict"), proportion_match.get("verdict")]
    if "fail" in verdicts:
        overall = "fail"
    elif "warn" in verdicts:
        overall = "warn"
    elif all(v == "pass" for v in verdicts):
        overall = "pass"
    else:
        overall = "skip"

    elapsed = time.time() - t0
    print(f"[DWPose] Validation done in {elapsed:.1f}s: pose={pose_match.get('verdict')}, proportions={proportion_match.get('verdict')}, overall={overall}", flush=True)

    return {
        "pose_match": pose_match,
        "proportion_match": proportion_match,
        "overall_verdict": overall,
        "validation_time": round(elapsed, 2),
    }
