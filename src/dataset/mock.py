"""Generate a mock dataset that goes through the exact same path as the real one:
empty placeholder .mp4 files in <data_dir>/{deception,truth}/ plus OpenFace-style
cache .pkl files named by feature_cache_path_for(). Deceptive clips get a weak,
learnable signal (shifted AUs, more gaze/pose motion) so training visibly works.
"""
import os

import numpy as np
import pandas as pd

from src.dataset.openface import DEFAULT_FEATURES, feature_cache_path_for


def generate_mock_dataset(data_dir, cache_dir, n_videos=80, fps=30, seed=0):
    rng = np.random.RandomState(seed)
    os.makedirs(cache_dir, exist_ok=True)
    for i in range(n_videos):
        deceptive = i % 2 == 0
        name = ("lie" if deceptive else "truth") + f"_{i:03d}"
        folder = os.path.join(data_dir, "deception" if deceptive else "truth")
        os.makedirs(folder, exist_ok=True)
        video_path = os.path.join(folder, name + ".mp4")
        open(video_path, "ab").close()

        # bimodal durations: short (~8 s) or long (~30 s) clips
        dur = rng.normal(9, 2) if rng.rand() < 0.5 else rng.normal(30, 3)
        n = max(int(dur * fps), 45)

        data = {}
        for c in DEFAULT_FEATURES:
            if c.endswith("_c"):
                data[c] = (rng.rand(n) < 0.3).astype(np.float32)
            elif c.startswith("AU"):
                data[c] = np.abs(rng.normal(0.8, 0.5, n)).astype(np.float32)
            elif c.startswith("pose_T"):
                data[c] = (np.cumsum(rng.normal(0, 0.3, n)) + rng.normal(0, 100)).astype(np.float32)
            elif c.startswith("pose_R") or c.startswith("gaze_angle"):
                data[c] = np.cumsum(rng.normal(0, 0.01, n)).astype(np.float32)
            else:
                data[c] = rng.normal(0, 0.2, n).astype(np.float32)

        if deceptive:  # weak signal
            for c in ("AU04_r", "AU12_r", "AU45_r", "AU23_r"):
                data[c] = data[c] + 0.4
            data["gaze_angle_x"] = data["gaze_angle_x"] + rng.normal(0, 0.05, n)
            data["pose_Ry"] = data["pose_Ry"] + rng.normal(0, 0.03, n)

        conf = np.clip(rng.normal(0.93, 0.04, n), 0, 1)
        for _ in range(rng.randint(0, 4)):  # tracking dropouts
            s = rng.randint(0, n)
            conf[s:s + rng.randint(5, 40)] = rng.uniform(0.1, 0.5)

        df = pd.DataFrame(data)
        df.insert(0, "frame", np.arange(1, n + 1))
        df.insert(1, "face_id", 0)
        df.insert(2, "timestamp", np.arange(n) / fps)
        df.insert(3, "confidence", conf)
        df.insert(4, "success", (conf > 0.3).astype(int))
        df.to_pickle(feature_cache_path_for(video_path, cache_dir))
    print(f"[mock] wrote {n_videos} placeholder videos + caches to {data_dir} / {cache_dir}")