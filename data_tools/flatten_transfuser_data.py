import os
import shutil

SRC_ROOT = "/home/niklas/adl-waypoint-net/transfuser_data"
DST_ROOT = "/home/niklas/adl-waypoint-net/dataset/train"

os.makedirs(DST_ROOT, exist_ok=True)

for dataset_name in os.listdir(SRC_ROOT):
    dataset_path = os.path.join(SRC_ROOT, dataset_name)
    if not os.path.isdir(dataset_path):
        continue
    for town_name in os.listdir(dataset_path):
        town_path = os.path.join(dataset_path, town_name)
        if not os.path.isdir(town_path):
            continue
        for route_name in os.listdir(town_path):
            route_path = os.path.join(town_path, route_name)
            dst_path = os.path.join(DST_ROOT, route_name)
            if os.path.exists(dst_path):
                print(f"Skipping {route_name}: already exists in destination.")
                continue
            print(f"Moving {route_path} -> {dst_path}")
            shutil.move(route_path, dst_path)
