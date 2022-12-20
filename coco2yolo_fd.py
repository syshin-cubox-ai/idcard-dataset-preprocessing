import json
import os
import shutil

import cv2
import numpy as np
import tqdm

img_dir = os.path.join('data', 'idcard_trainset')
annotation_path = os.path.join('data', 'idcard_trainset.json')
root = '../../data/idcard_fd'
dst_img_dir = os.path.join(root, 'train')

# Load coco format annotations
with open(annotation_path, 'r', encoding='utf-8') as f:
    coco = json.load(f)

# Map image id and image file name
image_id_filename = {image['id']: image['file_name'] for image in coco['images']}

# Copy yaml file
os.makedirs(dst_img_dir, exist_ok=True)
shutil.copy(os.path.join('yaml_files', 'idcard-fd.yaml'), root)

for anno in tqdm.tqdm(coco['annotations'], 'coco2yoloface'):
    # Convert annotation to yolo format
    category = anno['category_id'] - 1
    x1, y1, w, h = anno['bbox']
    cx = ((x1 + (x1 + w)) / 2) / anno['width']
    cy = ((y1 + (y1 + h)) / 2) / anno['height']
    w /= anno['width']
    h /= anno['height']
    landmarks = np.array(anno['segmentation'][0]).reshape(-1, 2)
    moments = cv2.moments(np.round(landmarks).astype(np.int32))
    mcx = round(moments['m10'] / moments['m00']) / anno['width']  # mass center x
    mcy = round(moments['m01'] / moments['m00']) / anno['height']  # mass center y
    landmarks = (landmarks / np.array([anno['width'], anno['height']])).reshape(-1).tolist()

    # Copy image corresponding to annotation
    image_filename = image_id_filename[anno['image_id']]
    shutil.copy(os.path.join(img_dir, image_filename),
                os.path.join(dst_img_dir, image_filename))

    # Save yolo format annotation
    anno_path = os.path.join(dst_img_dir, image_filename.replace('.jpg', '.txt'))
    with open(anno_path, 'w', encoding='utf-8') as f:
        f.write(f'{category} {cx} {cy} {w} {h}'
                f'{landmarks[6]} {landmarks[7]} {landmarks[4]} {landmarks[5]} '
                f'{landmarks[2]} {landmarks[3]} {landmarks[0]} {landmarks[1]} '
                f'{mcx} {mcy}')
