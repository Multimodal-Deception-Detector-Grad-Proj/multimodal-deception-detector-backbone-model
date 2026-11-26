# Visual EDA & Feature Extraction Pipeline (`src/eda/visual`)

## 1. System Overview

This package implements a modular, fault-tolerant, and disk-cached Exploratory Data Analysis (EDA) and preprocessing pipeline for multimodal deception detection. It transforms raw video datasets from Hugging Face into quality-audited per-scene metadata and cached `(N, T, D)` PyTorch feature tensors (facial landmarks, head pose, gaze, and frame-level emotion probabilities).

### Core Architectural Principles
* **Idempotency & Atomic Caching:** Expensive compute stages (`OpenFaceTableStore`, `SceneDetector`, `FeatureStore`) cache per-video artifacts to disk using atomic write-and-rename (`os.replace`) semantics. Interrupted runs can be safely resumed without corrupting cache files.
* **Parameter-Aware Cache Keys:** Downstream stages (`scenes.py`, `features.py`) hash configuration parameters into the cache filename (`params_key`), preventing stale artifacts when hyperparameters change.
* **Scene-Aware Quality Auditing:** Camera cuts in television clips (e.g., *Would I Lie to You?*) are detected *before* computing head-motion jitter so shot transitions are not falsely penalized as tracking instability.
* **Decoupled Gate Thresholding:** Raw quality metrics are computed once; pass/fail boolean gates (`QualityThresholds`) are applied dynamically in memory.
* **Separation of Visualization:** All Matplotlib/Seaborn rendering is isolated in `plots.py`, keeping pipeline stages purely data-driven.

---

## 2. Pipeline Execution Flow

```text
[Hugging Face Hub]
        │
        ▼  (Stage 0: dataset.py)
  HFDatasetLoader ──► VideoCatalog ──► meta_df (DataFrame)
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼  (Stage 1: openface_store.py)   ▼  (Stage 2: scenes.py)           ▼  (chunking.py)
  OpenFaceTableStore                SceneDetector                     ChunkPlanner
  (openface_cache/*.pkl)            (scene_cache_v2/*.json)           (plan_df, summary)
        │                                 │
        └────────────────┬────────────────┘
                         ▼  (Stage 3: quality.py)
                   QualityAuditor ──► quality_df (CSV) + QualityPlots
                         │
                         ▼  (Stage 4: features.py)
                   FeatureStore ────► FeatureSet (N, T, D) + EmotionPlots
                   (feature_cache/*.pt)
```

---

## 3. Module Reference & Contracts

### `config.py` — Centralized Configuration
* **`EDAConfig` (dataclass):** Single source of truth for paths, dataset repo IDs, worker counts, scene detection parameters, quality gate thresholds, sliding-window chunk sizes, and temporal sequence length (`target_t`).
* **Auto-derived Paths (`__post_init__`):** Automatically populates `openface_cache_dir`, `scene_cache_dir`, `feature_cache_dir`, and `quality_report_path` under `data_dir` if left as `None`.

### `caching.py` — Atomic On-Disk Persistence
* **Key Generators:**
  * `legacy_key(video_path) -> str`: `<stem>_<sha256(abspath)>`. Preserves backward compatibility with legacy OpenFace caches.
  * `params_key(video_path, params) -> str`: `<stem>_<sha256(abspath + sorted_params)[:16]>`. Invalidates cache hits automatically if parameters change.
* **`DiskCache`:**
  * Factory constructors: `DiskCache.pickle()`, `DiskCache.json()`, `DiskCache.torch()` (uses `weights_only=True`).
  * `has(video_path) -> bool`: Verifies file existence and non-zero byte size.
  * `load(video_path) -> Any | None`: Deserializes artifact; automatically deletes corrupted/unreadable files and returns `None` (cache miss).
  * `save(video_path, obj) -> None`: Writes to `<final>.tmp<pid>_<tid>` and atomically swaps via `os.replace`.

### `status.py` — Concurrency & Telemetry
* **`run_tasks(fn, tasks, workers=1, desc="", processes=False, on_error=None)`:**
  * `workers <= 1`: Runs sequentially in-process for clean stack traces.
  * `processes=False`: Uses `ThreadPoolExecutor` (ideal for subprocess-bound tasks like OpenFace CLI).
  * `processes=True`: Uses `ProcessPoolExecutor` with `multiprocessing.get_context("spawn")` to avoid OpenCV/PyTorch fork deadlocks in Jupyter notebooks.
  * Safely cancels pending futures on `KeyboardInterrupt`.
* **`StageReport` (dataclass):** Tracks `total`, `cached`, `computed`, `failed` (`List[Dict[str, str]]`), per-video `timings`, and `elapsed_sec`. Exposes `.summary()` and `.failures_df()`.

### `dataset.py` — Ingestion & Metadata Cataloging
* **`HFDatasetLoader`:** Lists and downloads `.mp4` files from `dandyidk/Multi-Modal-Lie-Detection` (`deception/`, `truth/` folders) via `huggingface_hub.snapshot_download`. Skips download if local files exist unless `force=True`.
* **`LabelResolver`:** Resolves ground truth (`Lie`, `Truth`, `Unknown`) by inspecting parent folder names first, falling back to whole-word regex token matching (`re.split(r"[^a-z]+", stem)`) so substrings like `"believe"` do not false-match `"lie"`.
* **`VideoCatalog`:** Iterates over local videos using `cv2.VideoCapture` to build `meta_df`.
  * **Primary Key:** `video_id` (POSIX path relative to dataset root, e.g., `deception/DOLOS/Female/Catherine/LS_WILTY_EP72_lie14.mp4`).
  * **Columns:** `sample_index`, `video_id`, `filename`, `path`, `label`, `label_source`, `frame_count`, `fps`, `width`, `height`, `duration_sec`.

### `openface_store.py` — Stage 1: OpenFace Extraction
* **`OpenFaceTableStore(extractor, cache_dir, workers=4, landmarks=True)`:**
  * Wraps an external `extractor.run_openface(path, tmp_dir, landmarks=...)` call inside a temporary directory.
  * Strips whitespace from OpenFace CSV column headers.
  * **Empty Table Caching:** Videos where OpenFace detects no face are cached as an empty `pd.DataFrame()` and logged in `StageReport.failed` so they are not repeatedly reprocessed on subsequent runs.
  * `ensure(catalog_df, verify=False) -> StageReport`: Computes uncached tables in parallel threads.

### `scenes.py` — Stage 2: Shot Boundary Detection
* **`SceneDetector`:** Produces 1-based inclusive frame intervals (`List[List[int]]`, e.g., `[[1, 120], [121, 300]]`) matching OpenFace's 1-indexed `frame` column.
* **Supported `method` modes:**
  1. `"content"` (default): Uses `scenedetect.ContentDetector` across `spawn` worker processes (`cv2.setNumThreads(1)` per worker to prevent CPU oversubscription). Cached via `params_key`.
  2. `"pose_jump"`: Zero-decode heuristic (`scenes_from_pose_jumps`) that triggers a scene cut when 3D head translation (`pose_Tx`, `pose_Ty`, `pose_Tz`) jumps by `> jump_mm` across consecutive tracked frames (`gap <= 5`).
  3. `"none"`: Treats each entire video as a single scene `[[1, frame_count]]`.

### `quality.py` — Stage 3: Per-Scene Quality Auditing
* **`QualityThresholds` (dataclass):** `conf_thresh=0.75`, `jitter_thresh=5.0` (mm), `max_dropout_pct=15.0` (%).
* **`QualityAuditor`:**
  * `audit_video(df, scenes) -> List[Dict]`: Slices OpenFace DataFrame by each scene's `[start, end]` frame bounds and computes:
    * `dropout_pct`: % of frames with `confidence < conf_thresh`.
    * `mean_confidence`: Mean OpenFace tracking confidence across the scene.
    * `j_tx`, `j_ty`, `j_tz`: Root-Mean-Square (RMS) frame-to-frame translation difference (mm) computed **strictly across adjacent valid frames** (`diff(frame) == 1`).
    * `head_pitch_std_deg`, `head_yaw_std_deg`, `head_roll_std_deg`: Standard deviation of head rotation (converted from radians to degrees).
    * `gaze_x_std_deg`, `gaze_y_std_deg`: Standard deviation of eye gaze angles (degrees).
  * `apply_gate(quality_df) -> pd.DataFrame`: Adds `max_jitter` and boolean `quality_gate_pass` (`dropout_pct <= max_dropout_pct` AND `max_jitter <= jitter_thresh`).

### `chunking.py` — Sliding-Window Sequence Projection
* **`ChunkPlanner(chunk_length_sec=8.0, overlap_sec=2.0)`:**
  * Calculates `stride_sec = chunk_length_sec - overlap_sec`.
  * `plan(meta_df) -> Tuple[pd.DataFrame, Dict]`: Appends `projected_chunks` per video and returns global sequence statistics (`target_fps`, `sequence_length_frames`, `projected_sequences`).

### `features.py` — Stage 4: Multimodal Feature Tensors
* **`FeatureSet` (dataclass):** Container holding stacked `features` tensor of shape `(N, T, D)`, `feature_names` (`List[str]`), `video_ids` (`List[str]`), and `labels` (`np.ndarray`).
  * `.emotion_labels`: Extracts names prefixed with `"emo_"`.
  * `.emotion_probs() -> np.ndarray`: Slices `(N, T, K)` emotion probability trajectories by feature name prefix.
* **`FeatureStore(extractor, tables, cache_dir, target_t=90)`:**
  * Inspects `extractor.extract_features` signature at init; if `openface_df` is accepted, passes the Stage 1 cached DataFrame to avoid running OpenFace a second time.
  * Cache key includes `target_t`, `feature_names`, `emotion_model.model_id`, `min_confidence`, `crop_faces`, `crop_margin`, and `scale_features`.
  * `ensure(catalog_df)` runs with `workers=1` (GPU-bound emotion inference).
  * `load_all(catalog_df) -> FeatureSet` concatenates cached `(1, T, D)` tensors into `(N, T, D)`.

### `plots.py` — Stateless Visualization Suite
All methods accept `show: bool = True` and return the `matplotlib.figure.Figure` object:
* **`DatasetPlots`:** `duration_and_chunks(plan_df, ...)`, `class_balance(meta_df)`.
* **`QualityPlots`:**
  * `dashboard(q, jitter_thresh)`: 2x2 grid (dropout histogram, confidence histogram, translation jitter boxplot with threshold line, gaze dispersion boxplot).
  * `gate(q)` & `gate_by_label(q)`: Overall pass/fail counts and class-stratified pass rates (detects label-imbalance bias introduced by filtering).
  * `unstable(q, n=10) -> pd.DataFrame`: Returns top-`n` worst scenes sorted by dropout and jitter.
* **`EmotionPlots`:**
  * `summary(probs, labels) -> Dict`: Computes global Shannon entropy (bits), max theoretical entropy, mean frame entropy, mean top-1 probability, and dominant emotion.
  * `global_means()`, `by_condition()`, `trajectory()`, `temporal_stability()` (measures L1 frame-to-frame prediction flicker $\frac{1}{T-1}\sum_t \|p_t - p_{t-1}\|_1$).

---

## 4. AI Assistant Context & Extension Guidelines

When modifying or extending this codebase, adhere to the following invariants:

1. **Primary Key Integrity:** Always join or index video records using `video_id` (relative POSIX path), **never** `filename` (basenames collide across `deception/` and `truth/` subdirectories).
2. **1-Based Frame Indexing:** Both OpenFace output tables and `SceneDetector` intervals (`[start, end]`) use **1-based inclusive** frame indexing. Do not convert to 0-based slicing without adjusting boundary checks in `QualityAuditor.audit_video`.
3. **Speaker-Disjoint Splits:** `video_id` paths embed speaker identities (e.g., `.../Female/<SpeakerName>/<Clip>.mp4`). Any downstream train/validation/test splitting logic must group by speaker identity to prevent data leakage.
4. **Multiprocessing Safety:** Any function passed to `run_tasks(..., processes=True)` **must** be defined at module top-level so Python's `spawn` context can pickle it.
5. **External Dependency Interface:** `OpenFaceTableStore` and `FeatureStore` expect an external `extractor` object implementing:
   * `extractor.run_openface(video_path: str, out_dir: str, landmarks: bool) -> pd.DataFrame`
   * `extractor.extract_features(video_path: str, target_t: int, openface_df: Optional[pd.DataFrame] = ...) -> torch.Tensor` (shape `(1, T, D)`)
   * `extractor.feature_names: Sequence[str]` (emotion channels prefixed with `"emo_"`)