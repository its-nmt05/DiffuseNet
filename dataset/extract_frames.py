import os
import argparse
import cv2


# extract frames from video with specified parameters
def extract_frames(input_video, output_dir, interval, resolution=None, start_sec=0.0):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) # get video fps
    if fps <= 0:
        fps = 30.0 # default to 30 fps
    interval_frames = int(round(interval * fps)) # convert interval to frames
    skip_frames = int(round(start_sec * fps)) # convert start time to frames

    os.makedirs(output_dir, exist_ok=True)

     # Skip initial frames
    for _ in range(skip_frames):
        cap.read()

    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break   # end

        # process frame
        if frame_idx % interval_frames == 0:
            if resolution:
                frame = cv2.resize(frame, (resolution, resolution),
                                   interpolation=cv2.INTER_AREA)
            filename = f"frame_{saved_idx:05d}.jpg"
            path = os.path.join(output_dir, filename)
            cv2.imwrite(path, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved_idx} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video at regular intervals")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--interval", "-t", type=float, default=1.0)
    parser.add_argument("--resolution", "-r", type=int, default=None)
    parser.add_argument("--skip", type=float, default=0.0)
    args = parser.parse_args()

    extract_frames(args.input, args.output, args.interval, args.resolution, args.skip)


if __name__ == "__main__":
    main()