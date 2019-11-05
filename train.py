"""
Retrain the YOLO model for your own dataset.
"""
import csv
import string
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

import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity


def get_session():
    ''' Set tf backend to allow memory to grow, instead of claiming everything '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
K.tensorflow_backend.set_session(get_session())


input_shape = (416, 416)  # multiple of 32, hw

vvc_dataset = {'train_file': 'tags/train.txt', 
               'val_file': 'tags/val.txt', 
               'classes_file': 'model_data/voc_classes.txt'}

coco_dataset = {'train_file': 'tags/coco_train2017.txt', 
                'val_file': 'tags/coco_val2017.txt', 
                'classes_file': 'model_data/coco_classes.txt'}

coco_dataset_small = {'train_file': 'tags/coco_train2017_10.txt', 
                      'val_file': 'tags/coco_val2017_10.txt', 
                      'classes_file': 'model_data/coco_classes.txt'}

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
    model_name = 'tiny-yolov3-pretrained'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/anchors/tiny-yolov3-transfer.txt'
    
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
        
    assert len(anchors)==6 # default setting
    
    model = create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True,
            freeze_body=2, weights_path='model_data/yolov3-tiny_weights.h5')
    
    training(model_name=model_name, 
             model=model, 
             classes_path=classes_path, 
             anchors_path=anchors_path, 
             frozen_epochs=0, 
             unfreeze_epochs=50)


def vvc_yolov3_training():
    model_name = 'vvc3-yolov3'
    dataset = coco_dataset_small
    classes_path = dataset['classes_file']
    anchors_path = 'model_data/anchors/coco_tiny-yolov3-transfer.txt'
    
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    
    assert len(anchors) == 6 # default setting
            
    model = create_vvc_model(yolo3_model.vvc3_yolo_body, input_shape, anchors, num_classes)
    
    training(model_name=model_name,
             model=model, 
             dataset=dataset, 
             anchors_path=anchors_path, 
             unfreeze_epochs=2)
    
        
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
            

def training(model_name, model, dataset, anchors_path, frozen_epochs=0, unfreeze_epochs=50):
    
    train_annotation_path = dataset['train_file']
    val_annotation_path = dataset['val_file']
    timestamp = datetime.now().strftime('%y%m%d_%H%M')
    log_dir = Path('logs/'+ model_name + ' {} f{:02d} u{:02d} /'.format(timestamp, frozen_epochs, unfreeze_epochs))
    log_dir.mkdir(exist_ok=True)
    model_folder = Path('model_data/' + model_name)
    model_folder.mkdir(exist_ok=True)
    total_epochs = frozen_epochs + unfreeze_epochs
   
    class_names = get_classes(dataset['classes_file'])
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    
    # Callbacks
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(str(log_dir.joinpath('epoch{epoch:02d}.h5')),
        monitor='val_loss', save_weights_only=True, save_best_only=True, 
        period=max(1, total_epochs//10))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    csv_logger = CSVLogger(str(model_folder.joinpath('training_log.csv')))

    # Train/Val split
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    
    num_val = len(val_lines)
    num_train = len(train_lines)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if frozen_epochs > 0:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        train_data_generator = data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes)
        val_data_generator = data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes)
        
        history = model.fit_generator(
            train_data_generator,
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=val_data_generator,
            validation_steps=max(1, num_val//batch_size),
            epochs=frozen_epochs,
            initial_epoch=0,
            callbacks=[logging, checkpoint, csv_logger])
        model.save_weights(log_dir.joinpath('weights_stage_1.h5'))

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if unfreeze_epochs > 0:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the {} layers.'.format(len(model.layers)))

        batch_size = 12 # (16) note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        train_data_generator = data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes)
        val_data_generator = data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes)
        
        history = model.fit_generator(
            train_data_generator,
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=val_data_generator,
            validation_steps=max(1, num_val//batch_size),
            epochs=total_epochs,
            initial_epoch=frozen_epochs,
            callbacks=[logging, checkpoint, csv_logger, reduce_lr, early_stopping])
        
        model.save_weights(log_dir.joinpath('weights.h5'))
    
    save_anchors(anchors, log_dir.joinpath('anchors.txt'))
    save_class_names(class_names, log_dir.joinpath('classes.txt'))
    
    # Further training if needed.
    
    plot_training_history(history, log_dir.joinpath('history.png'))
        
    
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


def save_class_names(class_names, path):
    with open(path, 'w') as file:
        for name in class_names:
            file.write(name + '\n')


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def save_anchors(anchors, path):
    anchors_list = anchors.reshape(1, -1).astype(int).tolist()
    with open(path, 'w') as file:
        line = ', '.join(str(i) for i in anchors_list)
        line = line.replace('[', '').replace(']', '')
        file.write(line)


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
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
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

def model_prunning():
    
    model_name = 'yolov3-prunned'
    classes_path = 'model_data/coco_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    
    train_annotation_path = 'tags/coco_train2017.txt'
    val_annotation_path = 'tags/coco_val2017.txt'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs/'+ model_name + ' {}/'.format(timestamp))
    log_dir.mkdir(exist_ok=True)
    model_folder = Path('model_data/' + model_name)
    model_folder.mkdir(exist_ok=True)
   
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    num_anchors = len(anchors)
    
    # Callbacks

    # Train/Val split
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    
    num_val = len(val_lines)
    num_train = len(train_lines)
    
    batch_size = 10
    
    K.clear_session() # get a new session
    
    # Prunning
    
    # Load the serialized model
    loaded_model = tf.keras.models.load_model('model_data/yolo_weights.h5', compile=False)
    
    loaded_model.summary()
    
    print('last output: ', loaded_model.layers[-1].output)
    
    #x.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    h, w = input_shape
    
    y_true = [tf.keras.Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5), name='input_t_{}'.format(l)) for l in range(3)]
    
    model_loss = tf.keras.layers.Lambda(yolo_loss, 
                                        output_shape=(1,), 
                                        name='yolo_loss',
                                        arguments={'anchors': anchors, 
                                                   'num_classes': num_classes, 
                                                   'ignore_thresh': 0.5}
                                        )([*loaded_model.output, *y_true])
                                        
    train_model = tf.keras.models.Model([loaded_model.input, *y_true], model_loss)
    
    train_model.summary()
    
        
    epochs = 2
    end_step = np.ceil(1.0 * num_train / batch_size).astype(np.int32) * epochs
    print(end_step)
    
    new_pruning_params = {
          'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.10,
                                                       final_sparsity=0.90,
                                                       begin_step=0,
                                                       end_step=end_step,
                                                       frequency=100)
    }
    
    new_pruned_model = sparsity.prune_low_magnitude(train_model, **new_pruning_params)
    
    #new_pruned_model.summary()
    
    print('last output: ', new_pruned_model.layers[-1].output)
    
    print('data: ', data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes))
    
    new_pruned_model.compile(optimizer='adam', loss={'prune_low_magnitude_yolo_loss': lambda y_true, y_pred: y_pred})
    
    # Add a pruning step callback to peg the pruning step to the optimizer's
    # step. Also add a callback to add pruning summaries to tensorboard
    callbacks = [
        sparsity.UpdatePruningStep(),
        sparsity.PruningSummaries(log_dir=log_dir, profile_batch=0)
    ]
    
    train_generator = data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes)
    val_generator = data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes)
    
    new_pruned_model.fit_generator(train_generator,
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=val_generator,
                validation_steps=max(1, num_val//batch_size),
                epochs=epochs,
                verbose=1,
                callbacks=callbacks
                )
    
    print('End training')
    
    score = new_pruned_model.evaluate_generator(val_generator, steps=num_val)
    print('Test loss:', score)
    
    # Export the pruned model
    
    # Remove prunning wrappers
    final_model = sparsity.strip_pruning(new_pruned_model)
    final_model.summary()
    
    final_model.save_weights(str(model_folder.joinpath('weights.h5').resolve()))
    
    for i, w in enumerate(final_model.get_weights()):
        print(
            "{} -- Total:{}, Zeros: {:.2f}%".format(
                final_model.weights[i].name, 
                w.size, np.sum(w == 0) / w.size * 100
            )
    )
        
    

    
def tf_yolo_loss(args, anchors):
    return anchors

if __name__ == '__main__':
    #yolov3_training()
    #tiny_yolov3_training()
    vvc_yolov3_training()
    #model_prunning()
