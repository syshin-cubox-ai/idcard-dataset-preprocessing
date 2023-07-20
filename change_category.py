import glob
import os

import tqdm

root = 'D:/data/passport_segment'
category_to_apply = 1

anno_paths = sorted(glob.glob(os.path.join(root, '**/*.txt'), recursive=True))
for anno_path in tqdm.tqdm(anno_paths):
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno = f.read()
    with open(anno_path, 'w', encoding='utf-8') as f:
        f.write(str(category_to_apply) + anno[1:])
