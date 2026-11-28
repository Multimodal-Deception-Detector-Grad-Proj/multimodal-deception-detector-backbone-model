import numpy as np
import torch
from transformers import AutoTokenizer
from src.common.constants import L_TXT, TARGET_T, TEXT_ENCODER


class TokenAligner:

    def __init__(self, model_name: str = TEXT_ENCODER, max_len: int = L_TXT, target_t: int = TARGET_T):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.target_t = target_t

    @staticmethod
    def _fill_missing_times(word_segments: list) -> list:
        """Untimed words (e.g. '2019') inherit prev word's end as start and the
        next timed word's start as end, instead of collapsing to frame 0."""
        words = []
        prev_end = 0.0
        for seg in word_segments:
            start, end = seg.get("start"), seg.get("end")
            words.append({"word": seg["word"].strip(), "start": start, "end": end, "_prev_end": prev_end})
            if end is not None:
                prev_end = end
        next_start = None
        for w in reversed(words):
            if w["start"] is None:
                w["start"] = w["_prev_end"]
                w["end"] = next_start if next_start is not None and next_start >= w["start"] else w["start"]
            elif w["end"] is None:
                w["end"] = w["start"]
            next_start = w["start"]
        return words

    def align_clip(self, word_segments: list, clip_fps: float) -> dict:
        token_ids, frame_spans = [], []

        token_ids.append(self.tokenizer.cls_token_id)
        frame_spans.append((0, 0))

        for seg in self._fill_missing_times(word_segments):
            if not seg["word"]:
                continue
            start_frame = int(np.floor(seg["start"] * clip_fps))
            end_frame = int(np.ceil(seg["end"] * clip_fps))

            # Words starting at/after the T-frame window are outside the shared clock: drop.
            if start_frame >= self.target_t:
                break
            end_frame = max(min(end_frame, self.target_t - 1), start_frame)

            for sub_id in self.tokenizer.encode(seg["word"], add_special_tokens=False):
                if len(token_ids) < self.max_len - 1:  # reserve slot for [SEP]
                    token_ids.append(sub_id)
                    frame_spans.append((start_frame, end_frame))
                else:
                    break
            if len(token_ids) >= self.max_len - 1:
                break

        token_ids.append(self.tokenizer.sep_token_id)
        last_frame = frame_spans[-1][1]
        frame_spans.append((last_frame, last_frame))

        seq_len = len(token_ids)
        pad_len = self.max_len - seq_len
        token_ids.extend([self.tokenizer.pad_token_id] * pad_len)
        frame_spans.extend([(0, 0)] * pad_len)
        text_mask = [1] * seq_len + [0] * pad_len

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.bool),
            "alignment_indices": torch.tensor(frame_spans, dtype=torch.long),
        }