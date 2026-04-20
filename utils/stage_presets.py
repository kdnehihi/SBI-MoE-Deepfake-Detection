"""Stage presets for staged MoE deepfake training."""

from __future__ import annotations

from utils.config import ModelConfig, StageConfig


def build_stage1_model_config() -> ModelConfig:
    config = ModelConfig()
    config.stage = StageConfig(
        name="stage1",
        enable_lora=True,
        enable_adapter=False,
        enable_moe_router=False,
        enable_classifier=True,
    )
    return config


def build_stage2_model_config() -> ModelConfig:
    config = ModelConfig()
    config.stage = StageConfig(
        name="stage2",
        enable_lora=True,
        enable_adapter=False,
        enable_moe_router=False,
        enable_classifier=True,
    )
    return config


def build_stage3_model_config() -> ModelConfig:
    config = ModelConfig()
    config.stage = StageConfig(
        name="stage3",
        enable_lora=True,
        enable_adapter=True,
        enable_moe_router=True,
        enable_classifier=True,
    )
    return config
