import json
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

coco_folder = "/home/juan/workspace/Maestria/Datasets/COCO/"
COCO_TRAIN_NAME = 'train2017'
COCO_TEST_NAME = 'val2017'
vvc_classes = {'bicycle': 1, 'car': 2, 'motorbike': 3, 'bus': 5, 'truck': 7}


def decode_category(category_id):
    if 1 <= category_id <= 11:
        return category_id - 1
    elif 13 <= category_id <= 25:
        return category_id - 2
    elif 27 <= category_id <= 28:
        return category_id - 3
    elif 31 <= category_id <= 44:
        return category_id - 5
    elif 46 <= category_id <= 65:
        return category_id - 6
    elif category_id == 67:
        return category_id - 7
    elif category_id == 70:
        return category_id - 9
    elif 72 <= category_id <= 82:
        return category_id - 10
    elif 84 <= category_id <= 90:
        return category_id - 11
    else:
        return category_id


def load_annotations(set_name):
    f = open(
        coco_folder + "annotations/instances_{}.json".format(set_name),
        encoding='utf-8')
    data = json.load(f)

    name_box_id = defaultdict(list)

    annotations = data['annotations']
    for annotation in annotations:
        image_id = annotation['image_id']
        image_path = coco_folder + '{}/{:012d}.jpg'.format(set_name, image_id)
        category_id = annotation['category_id']

        if Path(image_path).exists():
            category_id = decode_category(category_id)

            name_box_id[image_path].append([annotation['bbox'], category_id])

    annotation_per_line = defaultdict(list)

    for key in name_box_id.keys():
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])
            obj_class = int(info[1])

            annotation_per_line[key].append([x_min, y_min, x_max, y_max, obj_class])

    return annotation_per_line


def load_annotations_pandas(set_name):
    path = coco_folder + "annotations/instances_{}.json".format(set_name)
    f = open(path, encoding='utf-8')
    data = json.load(f)
    f.close()

    print('Json loaded', set_name)

    df = pd.DataFrame(data['annotations'])[['image_id', 'category_id', 'bbox']]

    df['category_id'] = df['category_id'].apply(lambda c: decode_category(c))

    df = df[df.apply(lambda row: row['category_id'] in vvc_classes.values(), axis=1)]

    print(df.info())
    print(df.head().to_string())

    stats = df.groupby(['image_id', 'category_id']).size().unstack(fill_value=0)

    stats = stats.sum(axis=0).rename({v: k for k, v in vvc_classes.items()})

    ax = stats.plot.bar(rot=0)
    fig = ax.get_figure()
    fig.savefig('tags/coco_{}.png'.format(set_name))
    fig.clf()

    print(stats.head().to_string())


def load_coco_tags(set_name='train2017', val_split=0, filter_classes=None):
    name_box_id = load_annotations(set_name)
    
    print("images: ", len(name_box_id))
    
    f = open('tags/coco_{}.txt'.format(set_name), 'w')

    for img_path, annotations in name_box_id.items():
        box_info = ''
        for annotation in annotations:
            x_min, y_min, x_max, y_max, obj_class = annotation
            if filter_classes is None or obj_class in filter_classes:
                box_info += " %d,%d,%d,%d,%d" % (x_min, y_min, x_max, y_max, obj_class)

        if len(box_info) > 0:
            line = str(img_path) + box_info + '\n'
            f.write(line)
            if random.random() < val_split:
                print('val_split')
    f.close()
    print('done')


def generate_cocov_train_val(val_split=0.2):
    """Generate the train and val set for COCOv"""

    name_box_id = load_annotations(COCO_TEST_NAME)


if __name__ == '__main__':
    generate_cocov_train_val()
    load_coco_tags('train2017', filter_classes=vvc_classes.values())
    load_coco_tags('val2017')
    load_annotations_pandas(COCO_TRAIN_NAME)
    load_annotations_pandas(COCO_TEST_NAME)
