"""Tests for configuration loading."""
import tempfile
import os
import yaml
from pathlib import Path
from config import Config, load_config, BehaviorConfig


def test_default_config():
    """Default config has expected values."""
    cfg = Config()
    assert cfg.behavior.head_down_penalty == 80
    assert cfg.attention_threshold == 50
    assert cfg.skip_frames == 2


def test_load_config_nonexistent_returns_defaults():
    """Loading missing file returns Config with defaults."""
    cfg = load_config("nonexistent_config_test.yaml")
    assert isinstance(cfg, Config)
    assert cfg.behavior.head_down_threshold == 0.08


def test_load_config_partial_override():
    """Partial YAML overrides only specified fields."""
    data = {"scoring": {"attention_threshold": 60}}
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    try:
        yaml.dump(data, tmp)
        tmp.close()
        cfg = load_config(tmp.name)
        assert cfg.attention_threshold == 60
        assert cfg.behavior.head_down_penalty == 80  # unchanged default
    finally:
        os.unlink(tmp.name)


def test_behavior_config_defaults():
    """BehaviorConfig defaults are correct."""
    b = BehaviorConfig()
    assert b.head_down_duration == 3.0
    assert b.eye_closed_duration == 2.0
    assert b.stillness_penalty == 50
    assert b.short_head_down_penalty == 30


def test_config_uses_yaml_overrides():
    """All behavior fields can be overridden from YAML."""
    data = {
        "behavior": {
            "head_down": {"threshold": 0.05, "penalty": 90},
            "eye_closed": {"ear_threshold": 0.15, "duration_sec": 3.0},
            "stillness": {"penalty": 40},
        }
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    try:
        yaml.dump(data, tmp)
        tmp.close()
        cfg = load_config(tmp.name)
        assert cfg.behavior.head_down_threshold == 0.05
        assert cfg.behavior.head_down_penalty == 90
        assert cfg.behavior.eye_closed_threshold == 0.15
        assert cfg.behavior.eye_closed_duration == 3.0
        assert cfg.behavior.stillness_penalty == 40
    finally:
        os.unlink(tmp.name)
