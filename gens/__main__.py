from __future__ import annotations
import argparse, json, glob, os
from .inference import setup_model, gens_frame_sampler  # adjust names if different

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GenS frame sampling (CLI wrapper for gens.inference)"
    )
    parser.add_argument("--model_id", default="yaolily/GenS")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    model, tokenizer, processor = setup_model(args.model_id)
    frame_paths = sorted(
        glob.glob(os.path.join(args.video_path, "*.png")) +
        glob.glob(os.path.join(args.video_path, "*.jpg")) +
        glob.glob(os.path.join(args.video_path, "*.jpeg"))
    )
    result = gens_frame_sampler(args.question, frame_paths, model, tokenizer, processor)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()