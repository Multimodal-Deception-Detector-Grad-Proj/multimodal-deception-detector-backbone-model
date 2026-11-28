from pathlib import Path
import pytest
import yaml

CONFIG_PATH = Path("configs/base_config.yaml")

def test_base_config_structure():
    assert CONFIG_PATH.exists(), f"Config file not found at {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    assert "transcribe" in config, "Missing 'transcribe' section in base_config.yaml"
    cfg = config["transcribe"]
    
    required_keys = [
        "model_size", "language", "device", "compute_type", 
        "batch_size", "beam_size", "condition_on_previous_text", 
        "no_speech_threshold", "cache_dir"
    ]
    for key in required_keys:
        assert key in cfg, f"Missing key '{key}' in transcribe configuration"

    assert cfg["language"] == "en"
    assert isinstance(cfg["batch_size"], int)