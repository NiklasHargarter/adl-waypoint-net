import os
import random
import shutil

DATASET_ROOT = "/home/niklas/adl-waypoint-net/dataset/train"
VAL_ROOT = "/home/niklas/adl-waypoint-net/dataset/val"
TEST_ROOT = "/home/niklas/adl-waypoint-net/dataset/test"

VAL_RATIO = 0.15  # 15% for validation
TEST_RATIO = 0.15  # 15% for test

random.seed(42)  # For reproducibility

# List all route folders
routes = [
    d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))
]
random.shuffle(routes)

total = len(routes)
val_count = int(total * VAL_RATIO)
test_count = int(total * TEST_RATIO)

val_routes = routes[:val_count]
test_routes = routes[val_count : val_count + test_count]
train_routes = routes[val_count + test_count :]

# Create destination folders
os.makedirs(VAL_ROOT, exist_ok=True)
os.makedirs(TEST_ROOT, exist_ok=True)

# Save split lists
with open("train_split.txt", "w") as f:
    for r in train_routes:
        f.write(r + "\n")
with open("val_split.txt", "w") as f:
    for r in val_routes:
        f.write(r + "\n")
with open("test_split.txt", "w") as f:
    for r in test_routes:
        f.write(r + "\n")

# Move folders
for r in val_routes:
    src = os.path.join(DATASET_ROOT, r)
    dst = os.path.join(VAL_ROOT, r)
    print(f"Moving {src} -> {dst}")
    shutil.move(src, dst)
for r in test_routes:
    src = os.path.join(DATASET_ROOT, r)
    dst = os.path.join(TEST_ROOT, r)
    print(f"Moving {src} -> {dst}")
    shutil.move(src, dst)

print(
    f"Done! {len(train_routes)} train, {len(val_routes)} val, {len(test_routes)} test routes."
)
