import json
import os
import shutil

import numpy as np
import tqdm

img_dir = os.path.join('data', 'idcard_trainset')
annotation_path = os.path.join('data', 'idcard_trainset.json')
root = '../../data/idcard_segment'
dst_img_dir = os.path.join(root, 'train')

# Load coco format annotations
with open(annotation_path, 'r', encoding='utf-8') as f:
    coco = json.load(f)

# Map image id and image file name
image_id_filename = {image['id']: image['file_name'] for image in coco['images']}

# Copy yaml file
os.makedirs(dst_img_dir, exist_ok=True)
shutil.copy(os.path.join('yaml_files', 'idcard-seg.yaml'), root)

for anno in tqdm.tqdm(coco['annotations'], 'coco2yolo-seg'):
    # Convert annotation to yolo format
    category = anno['category_id'] - 1
    segmentation = np.array(anno['segmentation'][0]).reshape(-1, 2)
    segmentation = (segmentation / np.array([anno['width'], anno['height']])).reshape(-1).tolist()

    # Copy image corresponding to annotation
    image_filename = image_id_filename[anno['image_id']]
    shutil.copy(os.path.join(img_dir, image_filename),
                os.path.join(dst_img_dir, image_filename))

    # Save yolo format annotation
    anno_path = os.path.join(dst_img_dir, image_filename.replace('.jpg', '.txt'))
    with open(anno_path, 'w', encoding='utf-8') as f:
        f.write(f'{category} {segmentation[0]:.4f} {segmentation[1]:.4f} {segmentation[2]:.4f} {segmentation[3]:.4f} '
                f'{segmentation[4]:.4f} {segmentation[5]:.4f} {segmentation[6]:.4f} {segmentation[7]:.4f}')
