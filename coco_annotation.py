import json
from pathlib import Path
from collections import defaultdict

coco_folder = "/home/juan/workspace/Maestria/Datasets/COCO/"
vvc_classes = [1, 2, 3, 5, 7]

def load_coco_tags(set_name='train2017'):
    name_box_id = defaultdict(list)
    id_name = dict()
    f = open(
        coco_folder + "annotations/instances_{}.json".format(set_name),
        encoding='utf-8')
    data = json.load(f)
    
    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = coco_folder + '{}/{:012d}.jpg'.format(set_name, id)
        cat = ant['category_id']
        
        if Path(name).exists():
            if cat >= 1 and cat <= 11:
                cat = cat - 1
            elif cat >= 13 and cat <= 25:
                cat = cat - 2
            elif cat >= 27 and cat <= 28:
                cat = cat - 3
            elif cat >= 31 and cat <= 44:
                cat = cat - 5
            elif cat >= 46 and cat <= 65:
                cat = cat - 6
            elif cat == 67:
                cat = cat - 7
            elif cat == 70:
                cat = cat - 9
            elif cat >= 72 and cat <= 82:
                cat = cat - 10
            elif cat >= 84 and cat <= 90:
                cat = cat - 11
            
            name_box_id[name].append([ant['bbox'], cat])
    
    print("images: ", len(name_box_id.keys()))
    
    f = open('tags/coco_{}.txt'.format(set_name), 'w')
    for key in name_box_id.keys():
        
        f.write(key)
        
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
                f.write(box_info)
        f.write('\n')
    f.close()
    
if __name__ == '__main__':
    load_coco_tags('train2017')
    load_coco_tags('val2017')

