import glob
import json
import os
import shutil

import numpy as np
import tqdm

src_dir = os.path.join('data', 'IDcard_Detection')
root = '../../data/IDCard_Segmentation'
dst_dir = os.path.join(root, 'all')
annotation_path = os.path.join('data', 'IDcard_Detection.json')

# Load coco format annotations
with open(annotation_path, 'r', encoding='utf-8') as f:
    coco = json.load(f)

# Check the number of images and number of annotations
num_images = len(glob.glob(os.path.join(src_dir, '*.jpg')))
assert num_images == len(coco['images']) == len(coco['annotations'])
print(f'전체 개수: {num_images}')

# Map image id and image file name
image_id_filename = {image['id']: image['file_name'] for image in coco['images']}

# Copy yaml file
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(os.path.join('yaml_files', 'idcard-seg.yaml'), root)

for anno in tqdm.tqdm(coco['annotations'], 'coco2yolo-seg'):
    # Check annotation
    assert len(anno['segmentation']) == 1, 'polygon이 1개가 아닙니다.'
    assert len(anno['segmentation'][0]) == 8, 'polygon의 점이 4개가 아닙니다.'
    assert anno['category_id'] == 1, '카테고리가 1개가 아닙니다.'
    assert len(anno['bbox']) == 4, 'bbox가 1개가 아닙니다.'

    # Convert annotation to yolo format
    category = anno['category_id'] - 1
    segmentation = anno['segmentation'][0]
    segmentation = (np.array(segmentation).reshape(-1, 2) / np.array([anno['width'], anno['height']])).reshape(-1).tolist()

    # Copy image corresponding to annotation
    image_filename = image_id_filename[anno['image_id']]
    shutil.copy(os.path.join(src_dir, image_filename),
                os.path.join(dst_dir, image_filename))

    # Save yolo format annotation
    anno_path = os.path.join(dst_dir, image_filename.replace('.jpg', '.txt'))
    with open(anno_path, 'w', encoding='utf-8') as f:
        f.write(f'{category} {segmentation[0]} {segmentation[1]} {segmentation[2]} {segmentation[3]} '
                f'{segmentation[4]} {segmentation[5]} {segmentation[6]} {segmentation[7]}')
