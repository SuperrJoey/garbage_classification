import os
import shutil

source = "dataset"   # change this!
target = "prepared_dataset"

mapping = {
    "plastic": "plastic",
    "paper": "paper",
    "hazardous": "hazardous"
}

for src, dst in mapping.items():
    src_path = os.path.join(source, src)
    dst_path = os.path.join(target, dst)

    os.makedirs(dst_path, exist_ok=True)

    for file in os.listdir(src_path):
        shutil.copy(os.path.join(src_path, file), os.path.join(dst_path, file))

print("Done!")