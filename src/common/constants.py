import os
TARGET_T = 200          # 200 frames @ 25 FPS = 8.0s (90th percentile coverage)
TARGET_FPS = 25.0

D_VIS = 51  # 49 OpenFace + valid + confidence; no dummy emotion features
OPENFACE_BIN = os.environ.get("OPENFACE_BIN", "FeatureExtraction")
D_TXT = 768          
L_TXT = 56           
TEXT_ENCODER = "microsoft/deberta-v3-base"
