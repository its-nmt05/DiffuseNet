import os
import random

frames_dir = r"D:\DiffuseNet\dataset\frames_64"
test_ratio = 0.2
seed = 42
train_out = "train_indices.txt"
test_out = "test_indices.txt"

def main():
    all_frames = sorted(
        f for f in os.listdir(frames_dir)
        if os.path.isfile(os.path.join(frames_dir, f))
    )
    total = len(all_frames)
    if total == 0:
        print("No frames found in", frames_dir)
        return

    # shuffle and split
    random.seed(seed)
    indices = list(range(total))
    random.shuffle(indices)
    test_count = int(total * test_ratio)
    test_indices = sorted(indices[:test_count])
    train_indices = sorted(indices[test_count:])

    with open(train_out, "w") as f:
        for idx in train_indices:
            f.write(f"{all_frames[idx]}\n")

    with open(test_out, "w") as f:
        for idx in test_indices:
            f.write(f"{all_frames[idx]}\n")

    print(f"Total frames: {total}")
    print(f"Train: {len(train_indices)} -> {train_out}")
    print(f"Test:  {len(test_indices)} -> {test_out}")

if __name__ == "__main__":
    main()
