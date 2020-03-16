import json
import random
from pathlib import Path
from collections import defaultdict

coco_folder = "/home/juan/workspace/Maestria/Datasets/COCO/"
vvc_classes = [1, 2, 3, 5, 7]


def load_coco_tags(set_name='train2017', val_split=0):
    name_box_id = defaultdict(list)
    f = open(
        coco_folder + "annotations/instances_{}.json".format(set_name),
        encoding='utf-8')
    data = json.load(f)
    
    annotations = data['annotations']
    for ant in annotations:
        image_id = ant['image_id']
        name = coco_folder + '{}/{:012d}.jpg'.format(set_name, image_id)
        cat = ant['category_id']
        
        if Path(name).exists():
            if 1 <= cat <= 11:
                cat = cat - 1
            elif 13 <= cat <= 25:
                cat = cat - 2
            elif 27 <= cat <= 28:
                cat = cat - 3
            elif 31 <= cat <= 44:
                cat = cat - 5
            elif 46 <= cat <= 65:
                cat = cat - 6
            elif cat == 67:
                cat = cat - 7
            elif cat == 70:
                cat = cat - 9
            elif 72 <= cat <= 82:
                cat = cat - 10
            elif 84 <= cat <= 90:
                cat = cat - 11
            
            name_box_id[name].append([ant['bbox'], cat])
    
    print("images: ", len(name_box_id.keys()))
    
    f = open('tags/coco_{}.txt'.format(set_name), 'w')

    for key in name_box_id.keys():
        
        line = str(key)
        img_with_vehicles = False
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])
            obj_class = int(info[1])
    
            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, obj_class)
            
            if obj_class in vvc_classes:
                img_with_vehicles = True
                line += box_info
        
        line += '\n'
        
        if img_with_vehicles:
            f.write(line)
            if random.random() < val_split:
                print('val_split')
    f.close()


if __name__ == '__main__':
    load_coco_tags('train2017')
    load_coco_tags('val2017')
