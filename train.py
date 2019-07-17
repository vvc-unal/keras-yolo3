"""
Retrain the YOLO model for your own dataset.
"""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
import matplotlib.pyplot as plt

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

import yolo3.model as yolo3_model


input_shape = (416, 416)  # multiple of 32, hw

def yolov3_training():
    model_name = 'yolov3-transfer'
    classes_path = 'model_data/coco_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
        
    model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze
    
    training(model_name=model_name, 
             model=model, 
             classes_path=classes_path, 
             anchors_path=anchors_path, 
             frozen_epochs=50, 
             unfreeze_epochs=50)
    
def tiny_yolov3_training():
    model_name = 'tiny-yolov3'
    classes_path = 'model_data/coco_classes.txt'
    anchors_path = 'model_data/tiny_yolo_anchors.txt'
    
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
        
    assert len(anchors)==6 # default setting
    
    model = create_tiny_model(input_shape, anchors, num_classes, load_pretrained=False,
            freeze_body=2, weights_path='model_data/yolov3-tiny_weights.h5')
    
    training(model_name=model_name, 
             model=model, 
             classes_path=classes_path, 
             anchors_path=anchors_path, 
             frozen_epochs=0, 
             unfreeze_epochs=2)

def vvc_yolov3_training():
    model_name = 'vvc2-yolov3'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/anchors/tiny-yolov3-transfer.txt'
    
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    
    assert len(anchors)==6 # default setting
            
    model = create_vvc_model(yolo3_model.vvc2_yolo_body, input_shape, anchors, num_classes)
    
    training(model_name=model_name,
             model=model, 
             classes_path=classes_path, 
             anchors_path=anchors_path, 
             unfreeze_epochs=5)
    
        
def read_training_log(model_name):
    model_folder = Path('model_data/' + model_name)
    log_file = model_folder.joinpath('training_log.csv')
    
    training_log = {
            'model_name' : model_name,
            'epoch': [],
            'loss': [],
            'val_loss': [],
        }
    
    with open(log_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            training_log['epoch'].append(row[0])
            training_log['loss'].append(row[1])
            training_log['val_loss'].append(row[2])
            
    return training_log
            

def training(model_name, model, classes_path, anchors_path, frozen_epochs=0, unfreeze_epochs=50):
    
    annotation_path = 'tags/train.txt'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs/'+ model_name + ' f{} u{} {}/'.format(frozen_epochs, unfreeze_epochs,timestamp))
    log_dir.mkdir(exist_ok=True)
    model_folder = Path('model_data/' + model_name)
    model_folder.mkdir(exist_ok=True)
   
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    
    # Callbacks
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(str(log_dir.joinpath('ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')),
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    csv_logger = CSVLogger(str(model_folder.joinpath('training_log.csv')))

    # Train/Val split
    val_split = 0.2
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if frozen_epochs > 0:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=frozen_epochs,
                initial_epoch=0,
                callbacks=[logging, checkpoint, csv_logger])
        model.save_weights(log_dir.joinpath('trained_weights_stage_1.h5'))

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if unfreeze_epochs > 0:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the {} layers.'.format(len(model.layers)))

        batch_size = 16 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=frozen_epochs + unfreeze_epochs,
            initial_epoch=frozen_epochs,
            callbacks=[logging, checkpoint, csv_logger, reduce_lr, early_stopping])
        
        model.save_weights(log_dir.joinpath('trained_weights_final.h5'))
    
   
    model.save_weights(model_folder.joinpath('weights.h5'))

    # Further training if needed.
    
    plot_training_history(history, model_folder.joinpath('history.png'))
        
    
def plot_training_history(history, save_path):
    # Plot training & validation loss values
   
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(history.epoch, history.history['loss'], label='Train')
    ax1.plot(history.epoch, history.history['val_loss'], label='Val')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='lower left')
    
    ax2.plot(history.history['lr'], 'k--')
    ax2.set_ylabel('Learning rate')
    ax2.legend(['learning rate'], loc='upper right')
    
    plt.title('Model loss')
    plt.tight_layout()

    plt.savefig(save_path)
    
    plt.show()

def plot_training_log():
    # Plot training & validation loss values
   
    log = read_training_log('tiny-yolov3')
    log2 = read_training_log('vvc2-yolov3')
    
    fig, ax1 = plt.subplots()
    
    ax1.plot(log['epoch'], log['loss'], label=(log['model_name']+'_Train'))
    ax1.plot(log['epoch'], log['val_loss'], label=(log['model_name']+'_Val'))
    ax1.plot(log2['epoch'], log2['loss'], label=(log2['model_name']+'_Train'))
    ax1.plot(log2['epoch'], log2['val_loss'], label=(log2['model_name']+'_Val'))
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='lower left')
    
    plt.title('Model loss')
    plt.tight_layout()
    
    plt.show()


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_vvc_model(body_func, input_shape, anchors, num_classes):
    '''create the training model, for VVC YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = body_func(image_input, num_anchors//2, num_classes)
    print('Create VVC YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    
    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    vvc_yolov3_training()
    plot_training_log()
