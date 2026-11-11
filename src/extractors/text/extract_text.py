import logging
from pathlib import Path
from typing import Dict, Union

import torch
from transformers import AutoModel

from src.common.constants import D_TXT, L_TXT, TARGET_FPS, TEXT_ENCODER
from src.common.contracts import check_text_sample, verify_text_tensors
from src.extractors.text.text_aligner import TokenAligner
from src.extractors.text.transcribe import Transcriber

logger = logging.getLogger(__name__)


class TextFeatureExtractor:

    def __init__(self, cfg: dict):
        self.cfg = cfg

        # Resolve compute device
        raw_device = cfg.get("transcribe", {}).get("device", "auto")
        if raw_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
        else:
            self.device = raw_device

        
        logger.info("Initializing Transcriber...")
        transcribe_cfg = cfg.get("transcribe", {})
        self.transcriber = Transcriber(transcribe_cfg) # takes configs from config file and initializes the transcriber using them

        logger.info("Initializing Token Aligner...")
        text_cfg = cfg.get("text_encoder", {})
        model_name = text_cfg.get("model_name", TEXT_ENCODER)
        max_len = text_cfg.get("max_len", L_TXT)
        self.aligner = TokenAligner(model_name=model_name, max_len=max_len)

        #Load DeBERTa Neural Encoder onto GPU/CPU
        logger.info(f"Loading DeBERTa model ({model_name}) onto {self.device}...")
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()

    @torch.no_grad() 
    def process_clip(
        self,
        clip_id: str,
        media_path: Union[str, Path],
        clip_fps: float = TARGET_FPS,
    ) -> Dict[str, torch.Tensor]:
        
        media_path = Path(media_path)

        # Transcribe Audio using WhisperX 
        transcribe_out = self.transcriber.transcribe_clip(clip_id, media_path)
        word_segments = transcribe_out.get("words", [])

        # Align Tokens to Discrete Video Frames
        aligned = self.aligner.align_clip(
            word_segments=word_segments, clip_fps=clip_fps
        )

        input_ids_batch = aligned["input_ids"].unsqueeze(0).to(self.device)
        text_mask_batch = aligned["text_mask"].unsqueeze(0).to(self.device)

        # Run DeBERTa Model Forward Pass
        outputs = self.encoder(
            input_ids=input_ids_batch,
            attention_mask=text_mask_batch,
        )

        # Extract last hidden state sequence [1, 56, 768] -> [56, 768] on CPU
        text_feats = (
            outputs.last_hidden_state.squeeze(0).cpu().to(torch.float32)
        )

        sample = {
            "input_ids": aligned["input_ids"],
            "text_feats": text_feats,
            "text_mask": aligned["text_mask"],
            "alignment_indices": aligned["alignment_indices"],
        }

        # Enforce Contracts
        check_text_sample(sample)
        verify_text_tensors(sample["text_feats"], sample["alignment_indices"])

        return sample