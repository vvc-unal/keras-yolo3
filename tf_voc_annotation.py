import xml.etree.ElementTree as ET
from os import getcwd, environ, path


# TODO load sets and classes dynamically 
sets=['bicycle_train', 'bus_train', 'bicycle_val', 'bus_val', 'car_train', 'car_val', 'motorbike_train', 'motorbike_val']

sets_tm=['seg_train', 'seg_val', 'tu_llave_train', 'tu_llave_val']

data_set_folder = path.join(environ['HOME'], 'workspace/Maestria/Videos', 'tf_pascal_voc')

data_set_folder_tm = path.join(environ['HOME'], 'workspace/Maestria/Videos',
                            'TM/TrabajadoresYPolicias', 'tf_pascal_voc')

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

classes_tm = ["tu_llave", "seg"]

def convert_annotation(image_id, list_file):
    in_file = open(path.join(data_set_folder, 'Annotations/%s.xml'%(image_id)))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == '__main__':
    
    with open('tags/train.txt', 'w') as list_file:
        for image_set in sets:
            image_ids = open(path.join(data_set_folder, 'ImageSets/Main/%s.txt'%(image_set))).read().strip().split()
            image_ids = [id for id in image_ids if id not in['1', '-1'] ]
            for image_id in image_ids:
                list_file.write(path.join(data_set_folder,'JPEGImages/%s.jpg'%(image_id)))
                convert_annotation(image_id, list_file)
                list_file.write('\n')
    
    print("Complete")
