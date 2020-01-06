import unittest

import yolo

from train import coco_dataset, create_vvc_model, get_classes, get_anchors, input_shape, training


class TrainVVC(unittest.TestCase):

    epochs = 10

    def train_vvc_model(self, model_name, body_name):
        dataset = coco_dataset
        classes_path = dataset['classes_file']
        anchors_path = '../model_data/tiny_yolo_anchors.txt'

        class_names = get_classes(classes_path)
        num_classes = len(class_names)
        anchors = get_anchors(anchors_path)

        assert len(anchors) == 6  # default setting

        body_func, is_tiny_version = yolo.YOLO.bodies.get(body_name)

        model = create_vvc_model(body_func, input_shape, anchors, num_classes)

        training(model_name=model_name,
                 model=model,
                 dataset=dataset,
                 anchors_path=anchors_path,
                 unfreeze_epochs=self.epochs)

    def test_vvc1(self):
        model_name = 'vvc1-yolov3'
        self.train_vvc_model(model_name, model_name.split('-')[0])

    def test_vvc2(self):
        model_name = 'vvc2-yolov3'
        self.train_vvc_model(model_name, model_name.split('-')[0])

    def test_vvc3(self):
        model_name = 'vvc3-yolov3'
        self.train_vvc_model(model_name, model_name.split('-')[0])


if __name__ == '__main__':
    unittest.main()
