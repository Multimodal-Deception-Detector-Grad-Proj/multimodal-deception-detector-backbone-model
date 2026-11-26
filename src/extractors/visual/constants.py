"""OpenFace column definitions shared across the package."""

POSE = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
GAZE = [
    "gaze_0_x", "gaze_0_y", "gaze_0_z",
    "gaze_1_x", "gaze_1_y", "gaze_1_z",
    "gaze_angle_x", "gaze_angle_y",
]
AU_INTENSITY = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
    "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r",
    "AU25_r", "AU26_r", "AU45_r",
]
AU_PRESENCE = [
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c", "AU09_c",
    "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c", "AU20_c", "AU23_c",
    "AU25_c", "AU26_c", "AU28_c", "AU45_c",
]
OPENFACE_COLS = AU_INTENSITY + AU_PRESENCE + POSE + GAZE
QUALITY_COLS = ["valid", "confidence"]
LANDMARK_X = [f"x_{i}" for i in range(68)]
LANDMARK_Y = [f"y_{i}" for i in range(68)]