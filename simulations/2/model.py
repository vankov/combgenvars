from __future__ import print_function

import keras.layers
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
#from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#K.set_image_dim_ordering('th')

from keras.callbacks import Callback, LearningRateScheduler

import keras.applications.vgg16
import keras.applications.resnet50
import keras.applications.inception_v3

from keras.layers import GlobalAveragePooling2D

import numpy as np
import tensorflow as tf
import scipy.spatial.distance
from scipy.special import expit
import scipy.misc
from itertools import permutations
import sys
import csv
import re

from settings import Settings
   

from tools import print_VARS

class ProgressTracker(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
                    
    def report(self, (acc, cos_sim, task_acc, task_cos), v_results):
        loss = np.mean(self.losses)
        self.losses = []
        #(v_acc, v_cos_sim))
        
        print("{:>4}\tLoss: {:.5f}\tVARS:{:.2f}({:3f}) Task:{:.2f}({:.3f})".format(
                    self.epoch, 
                    loss,  
                    acc,
                    cos_sim,
                    task_acc,
                    task_cos
                ), end="")
        
        if (len(v_results)):
            print("\t", end="")
            
        data = [loss, acc, cos_sim, task_acc, task_cos]
        
        for test_i in v_results:
            (v_acc, v_cos_sim, v_task_acc, v_task_cos) = v_results[test_i]
            data += [test_i, v_acc, v_cos_sim, v_task_acc, v_task_cos]    
            print("\t{} VARS:{:.2f}({:.3f}) Task:{:.2f}({:.3f})".format(
                        test_i, 
                        v_acc, 
                        v_cos_sim,
                        v_task_acc, 
                        v_task_cos
                    ), end="")
            
        print("")
        
        if Settings.output_file_name:

            with open(Settings.output_file_name, "a") as f_h:
                f_h.write("\t".join(map(str, data)))
                f_h.write("\n")
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def __init__(self, max_epoch):
        Callback.__init__(self)
        self.epoch = 0
        self.max_epoch = max_epoch
        
class ModelBase(object):


    def save(self, filename):
        self._model.save(filename)

    def load_weights(self, filename):
        self._model.load_weights(filename);

    def image_preprocess(self, image):
        return image
    
    def test(self, data_gen, n_batches = 1):
        
        VARS_acc_total = 0.0
        VARS_cos_total = 0.0 
        task_acc_total = 0.0 
        task_cos_total = 0.0
        
        for i in range(n_batches):
            inputs, targets = data_gen.next()           
            
            if type(inputs) == np.ndarray:
                (acc, cos, task_acc, task_cos) = self.calc_accuracy((inputs, targets))            
            else:
                (acc, cos, task_acc, task_cos) = self.calc_accuracy((inputs[0], inputs[1], targets))            
                
            VARS_acc_total += acc
            VARS_cos_total += cos
            task_acc_total += task_acc
            task_cos_total += task_cos
            
        if n_batches > 0 :
            return (
                        VARS_acc_total / n_batches, 
                        VARS_cos_total / n_batches,
                        task_acc_total / n_batches, 
                        task_cos_total / n_batches
                    )
        else:
            return 0, 0, 0, 0
        
    def input_generator(self, img_generator):
        pass
            
    def make_data_set(self, img_generator, data_set_path, targets_file = "targets.txt"):
        data_set = img_generator.flow_from_directory(
                data_set_path,
                classes = ["images"],
                target_size = (Settings.image_w, Settings.image_h),
                batch_size = Settings.batch_size,
                class_mode = 'sparse')
        
        targets_data = {}
        with open("{}/{}".format(data_set_path, targets_file), 'r') as F:
            reader = csv.reader(F, delimiter='\t')
            for line in reader:
                targets_data[line[0]] = map(float, line[1:])               

                    
        target_dim = len(targets_data[targets_data.keys()[0]])

        data_set.classes = np.zeros(shape=(len(data_set.filenames), target_dim))
        for i in range(len(data_set.filenames)):            
            img_id = re.search('images\/(.+)\.png', data_set.filenames[i]).group(1)            
            data_set.classes[i] = targets_data[img_id]
            
        return data_set
        
    def train(self, train_set_path, validation_set_paths, targets_file = "targets.txt", epoch_init = 0):
        img_gen = ImageDataGenerator(
#                samplewise_std_normalization = True,
#                samplewise_center = True,
                preprocessing_function=self.image_preprocess
            )
            
        train_set = self.make_data_set(img_gen, train_set_path, targets_file)
        valid_set_gens = {}
        for valid_set_path in validation_set_paths:
            valid_set_gens[valid_set_path] = self.input_generator(
                    self.make_data_set(img_gen, valid_set_path, targets_file)
                    )            
        progress_tracker = ProgressTracker(Settings.n_training_epochs)
        max_v_acc = 0
                
        callbacks=[progress_tracker]
        if (epoch_init > 0):
            epoch_init -= 1            

        train_set_gen = self.input_generator(train_set)

        for e in range(epoch_init, Settings.n_training_epochs):            
            progress_tracker.set_epoch(e + 1)

            self._model.fit_generator(
                train_set_gen, 
                steps_per_epoch = Settings.n_training_batches_per_epoch, 
                epochs = 1, 
                verbose = 0,
                callbacks = callbacks
                )

            if ((e + 1) % Settings.l_rate_update_freq) == 0:
                lr = K.get_value(self._model.optimizer.lr)
                K.set_value(self._model.optimizer.lr, lr * Settings.l_rate_decay)
                print("Learning rate changed to {:.7f}".format(lr * Settings.l_rate_decay))

            (VARS_acc, VARS_cos, task_acc, task_cos) = self.test(
                    train_set_gen, 
                    n_batches = Settings.n_test_batches
                )
            
            v_results = {}

            for valid_set_path in validation_set_paths:
                (v_VARS_acc, v_VARS_cos, v_task_acc, v_task_cos) = self.test(
                            valid_set_gens[valid_set_path], 
                            n_batches = Settings.n_test_batches      
                        )
                v_results[valid_set_path] = (v_VARS_acc, v_VARS_cos, v_task_acc, v_task_cos)

                
            progress_tracker.report(
                        (VARS_acc, VARS_cos, task_acc, task_cos), 
                        v_results
                    )
            
            if Settings.save_model_current_state:
                self.save("{}.current.h5".format(type(self).__name__))
            
            if Settings.save_model_best_state:
                if v_results[validation_set_paths[0]] > max_v_acc:
                    max_v_acc = v_results[validation_set_paths[0]]
                    self.save("{}.best.h5".format(type(self).__name__)) 
        
    def build(self):
        raise("Method 'build' not implemented for {}".format(self.__class__.__name__))
    
    def summary(self):
        return self.__model.summary()
        
    def __init__(self):
        self.build()


class VARSModel(ModelBase):

    def input_generator(self, img_generator):

        while(True):
            img_input, target = img_generator.next()    
            
            if self.__use_VARS:
                target, rnd_state = self.recode_target(target)
                
                yield [img_input, rnd_state], target
            else:
                yield img_input, target
            
    def get_rnd_state(self):
        
        state = np.arange(Settings.n_tokens)
        
        while True:
            if Settings.randomize_tokens:
                np.random.shuffle(state)

            yield state

    def recode_target(self, target):
        state = np.zeros(shape=(target.shape[0], Settings.n_tokens), dtype = np.int)
        
        for i in range(target.shape[0]):
            
            state[i] = next(self.get_rnd_state())
            
            recoded_target = [
                    np.zeros(shape = (
                            Settings.n_tokens, 
                            Settings.VARS_sem_dim)
                    ),
                    np.zeros(shape = (
                            Settings.max_arity,
                            Settings.n_tokens, 
                            Settings.n_tokens)
                    )
                ]
            example_target = [
                    target[i, :Settings.n_tokens * Settings.VARS_sem_dim].reshape(
                        Settings.n_tokens, 
                        Settings.VARS_sem_dim
                    ),
                    target[i, Settings.n_tokens * Settings.VARS_sem_dim: -Settings.task_dim].reshape(
                        Settings.max_arity,
                        Settings.n_tokens, 
                        Settings.n_tokens
                    )
                ]

            recoded_target[0] = example_target[0][state[i]]
            for a in range(Settings.max_arity):
                recoded_target[1][a] = example_target[1][a][state[i]].swapaxes(0, 1)[state[i]].swapaxes(0, 1)


            target[i] = np.concatenate(
                                (
                                    recoded_target[0].flatten(), 
                                    recoded_target[1].flatten(),
                                    target[i, -Settings.task_dim:]
                                )
                            )
            
            
        state = state.astype(np.float32)
        
        return target, state
               
    def calc_accuracy(self, data):        
        
        if self.__use_VARS:
            (image_inputs, rnd_states, targets) = data
            inputs = [image_inputs, rnd_states]
        else:
            (image_inputs, targets) = data
            inputs = image_inputs
            
        outputs = self._model.predict(inputs, batch_size=len(image_inputs))
             
        sum_VARS_acc = 0.0
        sum_VARS_cos = 0.0
            
        sum_task_acc = 0.0
        sum_task_cos = 0.0
        
        for i in range(len(image_inputs)):
            
            task_target = targets[i, -Settings.task_dim:]
            task_output = outputs[i, -Settings.task_dim:]
            
            if sum(np.exp(task_output)) <> 0.0:
                task_output = np.exp(task_output) / sum(np.exp(task_output))
                            
            sum_task_acc += 1.0 if np.argmax(task_target) == np.argmax(task_output) else 0.0
            sum_task_cos += 1 - scipy.spatial.distance.cosine(task_target, task_output)

            if not self.__use_VARS:
                continue
            
            sem_target = \
                targets[i,:Settings.n_tokens * Settings.VARS_sem_dim].reshape(
                        Settings.n_tokens, 
                        Settings.VARS_sem_dim)
            struct_target = \
                targets[i,Settings.n_tokens * Settings.VARS_sem_dim: -Settings.task_dim].reshape(
                        Settings.max_arity, 
                        Settings.n_tokens, 
                        Settings.n_tokens)
                
            sem_output = outputs[i,:Settings.n_tokens * Settings.VARS_sem_dim].reshape(
                        Settings.n_tokens, 
                        Settings.VARS_sem_dim)
            struct_output = \
                outputs[i,Settings.n_tokens * Settings.VARS_sem_dim: -Settings.task_dim].reshape(
                        Settings.max_arity, 
                        Settings.n_tokens, 
                        Settings.n_tokens)
            
            
            sem_output = expit(sem_output)
            struct_output = expit(struct_output)
            
            sem_matches_all = 0
            struct_matches_all = 0
            sem_matches = 0
            struct_matches = 0
             

            for j in range(Settings.n_tokens):
                
                max_indices = sem_output[j].argsort()[::-1][0:int(np.sum(sem_target[j]))]

                    
                for k in max_indices:
                    sem_matches_all += 1
                    if sem_target[j][k]:
                        sem_matches += 1
                for arity in range(Settings.max_arity):                            
                    if np.sum(struct_target[arity, j]):
                        max_indices = struct_output[arity, j].argsort()[::-1][0:int(np.sum(struct_target[arity, j]))]
                        for k in max_indices:
                            struct_matches_all += 1
                            if struct_target[arity, j, k]:
                                struct_matches += 1         

            VARS_acc = (sem_matches == sem_matches_all) and (struct_matches == struct_matches_all)
                                         
            VARS_cos = (1 - Settings.sigma) * (1 - scipy.spatial.distance.cosine(
                    sem_output.flatten(), 
                    sem_target.flatten()))
            
            if (np.sum(struct_target.flatten())):
                VARS_cos += Settings.sigma * (1 - scipy.spatial.distance.cosine(
                        struct_output.flatten(), 
                        struct_target.flatten()))                
            else:
                VARS_cos += Settings.sigma

            if (Settings.print_error_and_stop) and not(VARS_acc):
                print(sem_matches, sem_matches_all)
                print(struct_matches, struct_matches_all)
                print(VARS_cos)                
                
                print_VARS((sem_target, struct_target))
                print_VARS((sem_output, struct_output), precision = 1)

                
                test_image = (np.transpose(image_inputs[i], [1, 2, 0]) * 255.).astype(np.int32)
                im = scipy.misc.toimage(test_image)
                im.save("error.png")
                
                exit(0)

                
            sum_VARS_acc += 1.0 if VARS_acc else 0.0
            sum_VARS_cos += VARS_cos
            
        return (
                    sum_VARS_acc / len(image_inputs), 
                    sum_VARS_cos / len(image_inputs),
                    sum_task_acc / len(image_inputs),
                    sum_task_cos / len(image_inputs)
                )
                          
    def build(self):
        def loss_f(y_true, y_pred):   

            loss = (1 if self.__use_VARS else 0) * tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=y_pred[:, :-Settings.task_dim], 
                            labels=y_true[:, :-Settings.task_dim]
                        )
                ) + tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                            logits=y_pred[:, -Settings.task_dim:], 
                            labels=y_true[:, -Settings.task_dim:]
                        )
                    )

            return loss
        
        image_input = Input(
                    shape = (Settings.image_w, Settings.image_h, 3, ),
                    dtype = 'float32',
                    name='image_input'
                )
                
        rnd_state_input = Input(
                    shape = (Settings.n_tokens, ),
                    dtype = 'float32', 
                    name='rnd_state'
                )
            
        conv1 = Conv2D(
                    filters = 8, 
                    kernel_size = (3, 3),
                    activation = "relu",
                    padding = 'same'
                )(image_input)
        
        max_pool_1 = MaxPooling2D(
                        pool_size = (2, 2)
                    )(conv1)
        
        conv2 = Conv2D(
                    filters = 16, 
                    kernel_size = (3, 3),
                    activation = "relu",
                    padding = 'same'
                )(max_pool_1)
        
        max_pool_2 = MaxPooling2D(
                        pool_size = (2, 2)
                    )(conv2)
        
        conv3 = Conv2D(
                    filters = 32, 
                    kernel_size = (3, 3),
                    activation = "relu",
                    padding = 'same'
                )(max_pool_2)
        
        max_pool_3 = MaxPooling2D(
                        pool_size = (2, 2)
                    )(conv3)
        
        flatten = Flatten() (max_pool_3)
                    
        
        if self.__use_VARS:
            fc_input = keras.layers.concatenate([flatten, rnd_state_input])
        else:
            fc_input = flatten
            
        fc1 = Dense(
                    units = 2 * 128,
                    activation = 'relu'
                    
                )(fc_input)
        
        fc2 = Dense(
                    units = 2 * 128,
                    activation = 'relu'
                )(fc1)
        
        output = Dense(
                    units = Settings.VARS_dim + Settings.task_dim
                )(fc2)
        
        if self.__use_VARS:            
            model_input = [image_input, rnd_state_input]
        else:
            model_input = [image_input]
            
            
        self._model = Model(
                    inputs = model_input, 
                    outputs = [output]
                )
                
        self._model.compile(
                    loss = loss_f, 
                    optimizer = RMSprop(lr = Settings.l_rate),
                    metrics=['cosine_proximity']
                )   
        
    def __init__(self, use_VARS = True):
        self.__use_VARS = use_VARS
        ModelBase.__init__(self)
                 
class VGGModel(ModelBase):

    def input_generator(self, img_generator):
        while(True):
            img_input, target = img_generator.next()    

            yield img_input, target[:,-Settings.task_dim:]
            
    def image_preprocess(self, image):
        image = np.expand_dims(image, axis = 0)
        image = keras.applications.vgg16.preprocess_input(image)
        return image[0]
    
    def calc_accuracy(self, data):
        (image_inputs, targets) = data
        outputs = self._model.predict(image_inputs, batch_size=len(image_inputs))
             
        task_acc_sum = 0.0
        task_cos_sum = 0.0
            
        for i in range(len(image_inputs)):
            task_acc_sum += np.argmax(targets[i]) == np.argmax(outputs[i])
            task_cos_sum += 1 - scipy.spatial.distance.cosine(
                            targets[i].flatten(), 
                            outputs[i].flatten()
                        )

        return (
                0.0, 
                0.0,
                task_acc_sum / len(image_inputs), 
                task_cos_sum / len(image_inputs),                
            )
    
    def build(self):
        
        vgg_model = keras.applications.vgg16.VGG16(
                    include_top = False,
                    input_shape = (Settings.image_w, Settings.image_h, 3),
                    weights = 'imagenet',
                    pooling = None
                    #'avg' #global average pooling
                )
        
        for layer in vgg_model.layers:
            layer.trainable = False
               
        if self.__useGAP:
            flatten = GlobalAveragePooling2D()(vgg_model.output)
        else:
            flatten = Flatten()(vgg_model.output)
            
        fc1 = Dense(units = 2 * 128, activation="relu")(flatten)
        fc2 = Dense(units = 2 * 128, activation="relu")(fc1)
        output = Dense(
                    units = Settings.task_dim,
                    activation = 'softmax'
                )(fc2)
        
        self._model = Model(
                    inputs = vgg_model.input,
                    outputs = [output]
                )
        
        self._model.compile(
                    loss = "categorical_crossentropy",
                    optimizer = RMSprop(lr = Settings.l_rate),
                    metrics = ['accuracy']
                )
        pass                
    
    def __init__(self, useGAP = False):
        self.__useGAP = useGAP
        ModelBase.__init__(self)