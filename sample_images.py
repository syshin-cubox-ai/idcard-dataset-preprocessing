import glob
import os
import shutil

import numpy as np
import tqdm

root = '../../data/sample'
result_dir = '../../data/extra_trainset'

hand_paths = np.array(sorted(glob.glob(os.path.join(root, 'hand', '*.jpg'))))
indoor_paths = np.array(sorted(glob.glob(os.path.join(root, 'indoor', '*.jpg'))))
outdoor_paths = np.array(sorted(glob.glob(os.path.join(root, 'outdoor', '*.jpg'))))

# Sample images
hand_paths = hand_paths[::2]
indoor_paths = indoor_paths.reshape(-1, 10)[:, [0, 3, 7]].flatten()
outdoor_paths = outdoor_paths[::3]

# Save sampled images
sampled_paths = hand_paths.tolist() + indoor_paths.tolist() + outdoor_paths.tolist()
os.makedirs(result_dir, exist_ok=True)
for src_path in tqdm.tqdm(sampled_paths, 'Save sampled images'):
    shutil.copy(src_path, os.path.join(result_dir, os.path.basename(src_path)))
