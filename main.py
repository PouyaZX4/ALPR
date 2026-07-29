import argparse
import yaml
from src.pipeline.pipeline import SpeedALPRPipeline
from src.utils.video_io import read_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline = SpeedALPRPipeline(config)

    for frame_idx, frame in enumerate(read_frames(args.video)):
        pipeline.process_frame(frame, frame_idx)

    print("Done. Results in", config["paths"]["database"])

if __name__ == "__main__":
    main()
