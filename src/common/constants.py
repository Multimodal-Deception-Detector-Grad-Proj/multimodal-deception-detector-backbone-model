TARGET_T = 200          # 200 frames @ 25 FPS = 8.0s (90th percentile coverage)
TARGET_FPS = 25.0

D_VIS = 58              # Action Units + Pose + Gaze + Emotion Probabilities + valid + confidence
OPENFACE_BIN = "/home/dandy/OpenFace/build/bin/FeatureExtraction"
D_TXT = 768          
L_TXT = 56           
TEXT_ENCODER = "microsoft/deberta-v3-base"

# ---------------------------------------------------------------------------
# Audio stream
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 16_000
assert TARGET_FPS.is_integer() and AUDIO_SAMPLE_RATE % int(TARGET_FPS) == 0, (
    "hop_length = sample_rate / FPS must be an integer, otherwise audio and video frames drift apart"
)
AUDIO_HOP_LENGTH = AUDIO_SAMPLE_RATE // int(TARGET_FPS)      # 640 samples = exactly one video frame (40 ms)
AUDIO_N_FFT = 1024                                           # STFT window: 64 ms, 15.6 Hz bins (resolves pitch harmonics)
AUDIO_WIN_LENGTH = AUDIO_N_FFT
AUDIO_F_MIN, AUDIO_F_MAX = 0.0, 8000.0
AUDIO_LOG_EPS = 1e-9
AUDIO_CLIP_SAMPLES = TARGET_T * AUDIO_HOP_LENGTH             # 128,000 samples = 8.0 s


# Frame-level channels appended after the mel bins (all aligned to the same T grid):
#   valid        1.0 on real frames, 0.0 on padding 
#   f0_semitone  absolute pitch in semitones re 27.5 Hz (NOT clip-normalized, so mean pitch level is kept)
#   voiced       1.0 where a pitch was detected
#   loudness_db  frame intensity in dB
#   hnr_db       harmonics-to-noise ratio in dB
AUDIO_EXTRA_CHANNELS = ("valid", "f0_semitone", "voiced", "loudness_db", "hnr_db")
D_AUD = 64                                                   # total audio channels (multiple of 8)
AUDIO_N_MELS = D_AUD - len(AUDIO_EXTRA_CHANNELS)             # 59 mel bins
AUDIO_CHANNEL_INDEX = {name: AUDIO_N_MELS + i for i, name in enumerate(AUDIO_EXTRA_CHANNELS)}