import json
import os

import tqdm

annotation_path = os.path.join('data', 'IDcard_Detection.json')

# Load coco format annotations
with open(annotation_path, 'r', encoding='utf-8') as f:
    coco = json.load(f)

# Map image id and image file name.
image_id_filename = {image['id']: image['file_name'] for image in coco['images']}

# Check annotations
multiple_polygon = []
invalid_num_polygon_points = []
for anno in tqdm.tqdm(coco['annotations'], 'Check dataset'):
    if not len(anno['segmentation']) == 1:
        multiple_polygon.append(image_id_filename[anno['image_id']])
    if not len(anno['segmentation'][0]) == 8:
        invalid_num_polygon_points.append(image_id_filename[anno['image_id']])

print(f'polygon이 여러 개: {multiple_polygon}')
print(f'polygon의 점이 잘못됨: {invalid_num_polygon_points}')
