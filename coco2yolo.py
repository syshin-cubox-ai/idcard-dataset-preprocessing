import json
import os
import shutil

import tqdm

root = '../../data/IDCard_Detection'

with open(os.path.join('data', 'IDcard_Detection.json'), 'r', encoding='utf-8') as f:
    coco = json.load(f)

# Map image id and image file name.
image_id_filename = {image['id']: image['file_name'] for image in coco['images']}

# Copy yaml file
os.makedirs(os.path.join(root, 'all'), exist_ok=True)
shutil.copy(os.path.join('data', 'idcard.yaml'), root)

for anno in tqdm.tqdm(coco['annotations'], 'coco2yolo'):
    # assert len(anno['segmentation']) == 1, 'polygon이 1개가 아닙니다.'
    # assert len(anno['segmentation'][0]) == 8, 'polygon의 점이 4개가 아닙니다.'
    assert anno['category_id'] == 1, '카테고리가 1개가 아닙니다.'
    assert len(anno['bbox']) == 4, 'bbox가 1개가 아닙니다.'

    # Convert annotation to yolo format
    category = anno['category_id'] - 1
    x1, y1, w, h = anno['bbox']
    cx = ((x1 + (x1 + w)) / 2) / anno['width']
    cy = ((y1 + (y1 + h)) / 2) / anno['height']
    w /= anno['width']
    h /= anno['height']

    # Copy image corresponding to annotation
    image_filename = image_id_filename[anno['image_id']]
    shutil.copy(os.path.join('data', 'IDCard_Detection', image_filename),
                os.path.join(root, 'all', image_filename))

    # Save yolo format annotation
    anno_path = os.path.join(root, 'all', image_filename.replace('.jpg', '.txt'))
    with open(anno_path, 'w', encoding='utf-8') as f:
        f.write(f'{category} {cx} {cy} {w} {h}')
