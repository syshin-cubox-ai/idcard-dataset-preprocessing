import glob
import os
import random
import shutil

import tqdm

root = '../../data/IDCard_Segmentation'
val_value = 0.1

all_images = glob.glob(os.path.join(root, 'all', '*.jpg'))
all_labels = glob.glob(os.path.join(root, 'all', '*.txt'))
assert len(all_images) == len(all_labels)
print(f'전체 개수: {len(all_images)}')

num_val = round(len(all_images) * val_value)

random.shuffle(all_images)
val_images = all_images[:num_val]
train_images = all_images[num_val:]
print(f'trainset 개수: {len(train_images)}')
print(f'valset 개수: {len(val_images)}')

shutil.rmtree(os.path.join(root, 'train'), ignore_errors=True)
shutil.rmtree(os.path.join(root, 'val'), ignore_errors=True)
os.makedirs(os.path.join(root, 'train'))
os.makedirs(os.path.join(root, 'val'))
for train_image in tqdm.tqdm(train_images, 'Split train'):
    shutil.move(train_image, train_image.replace('all', 'train'))
    train_label = train_image.replace('.jpg', '.txt')
    shutil.move(train_label, train_label.replace('all', 'train'))

for val_image in tqdm.tqdm(val_images, 'Split val'):
    shutil.move(val_image, val_image.replace('all', 'val'))
    val_label = val_image.replace('.jpg', '.txt')
    shutil.move(val_label, val_label.replace('all', 'val'))

shutil.rmtree(os.path.join(root, 'all'))
