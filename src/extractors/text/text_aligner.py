import numpy as np
import torch
from transformers import AutoTokenizer
from src.common.constants import L_TXT, TEXT_ENCODER


class TokenAligner:

    def __init__(self, model_name: str = TEXT_ENCODER, max_len: int = L_TXT):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def align_clip(self, word_segments: list, clip_fps: float) -> dict:

        token_ids = []
        frame_spans = []

        # Add [CLS] special token at position 0
        cls_id = self.tokenizer.cls_token_id
        token_ids.append(cls_id)
        frame_spans.append((0, 0))  # [CLS] spans start of video

        for segment in word_segments:
            word = segment["word"].strip()
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)

            # Convert seconds to discrete frame indices based on clip FPS
            start_frame = int(np.floor(start_time * clip_fps))
            end_frame = int(np.ceil(end_time * clip_fps))

            # DeBERTa tokenization (handles sub-word splitting)
            sub_tokens = self.tokenizer.encode(word, add_special_tokens=False)

            for sub_id in sub_tokens:
                if len(token_ids) < self.max_len - 1:  # Reserve slot for [SEP]
                    token_ids.append(sub_id)
                    frame_spans.append((start_frame, end_frame))
                else:
                    break
            
            if len(token_ids) >= self.max_len - 1:
                break

        # Add [SEP] special token at the end
        sep_id = self.tokenizer.sep_token_id
        token_ids.append(sep_id)
        last_frame = frame_spans[-1][1] if frame_spans else 0
        frame_spans.append((last_frame, last_frame))

        # Padding up to L_TXT = 56
        seq_len = len(token_ids)
        pad_len = self.max_len - seq_len

        pad_id = self.tokenizer.pad_token_id
        token_ids.extend([pad_id] * pad_len)
        frame_spans.extend([(0, 0)] * pad_len)  # Padded tokens span no frames

        text_mask = [1] * seq_len + [0] * pad_len

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.bool),
            "alignment_indices": torch.tensor(frame_spans, dtype=torch.long),  # Shape: [56, 2]
        }