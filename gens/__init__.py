from __future__ import annotations
from typing import List, Dict, Any
import glob, os

# Directly import from the relocated file
from .inference import setup_model, gens_frame_sampler  # adjust names if different

def run_gens(
    video_path: str,
    question: str,
    model_id: str = "yaolily/GenS",
) -> Dict[str, Any]:
    """
    High-level Python API for running GenS frame selection directly.
    Loads the model, collects frames in `video_path`, and runs gens_frame_sampler.
    Returns whatever gens_frame_sampler returns (usually a dict).
    """
    model, tokenizer, processor = setup_model(model_id)

    frame_paths: List[str] = sorted(
        glob.glob(os.path.join(video_path, "*.png")) +
        glob.glob(os.path.join(video_path, "*.jpg")) +
        glob.glob(os.path.join(video_path, "*.jpeg"))
    )
    return gens_frame_sampler(question, frame_paths, model, tokenizer, processor)

__all__ = ["run_gens", "setup_model", "gens_frame_sampler"]