from __future__ import print_function
import tensorflow as tf
import keras 
import numpy as np
from scipy import spatial
from settings import Settings
from sets import VARSEncoder
from scipy.special import expit

class LearningProgressTracker(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
                    
    def report(self, tests):
        if len(self.losses) > 0:
            loss = np.mean(np.array(self.losses))
        else:
            loss = 0
        
        self.losses = []
        
        if Settings.training_progress_report:
            print("{:>3} Loss: {:.12f}".format(
                        self.__trial_no, 
                        loss
                    ), end="")
    
            for test in tests:
                                    
                print("\t{}: {:.2f}({:.3f}) {:.2f}({:.3f})".format(
                            test,
                            tests[test][0],
                            tests[test][1],                      
                            tests[test][2],
                            tests[test][3],
                        ), end="")
            
            print("")            
        

    def set_trial(self, trial_no):
        self.__trial_no = trial_no
        
    def __init__(self):
        keras.callbacks.Callback.__init__(self)
        self.__trial_no = 0
        self.losses = []

class BaseModel:
    def print_summary(self):
        print(self._model.summary())

    def save_weights(self, training_trials = 0, state = "current"):
    
        weights_file = "{}.{}.{}.h5".format(
                self.__name,
                training_trials,
                state)
        self._model.save_weights(weights_file)        
        
    def load_weights(self, weights_file = False):
        if weights_file == False:
            weights_file = "{}.last.h5".format(self.__name)

        self._model.load_weights(weights_file)

    def _calc_progress(self, t_output, t_target):
        test_pos = Settings.n_roles

        # if t_target[test_pos][0] == 1 and t_input[test_pos][0] == 1:
        #     print(t_output[test_pos])
        
#        print(t_target)
#        print(t_output)
        acc = 1 if np.argmax(t_target[test_pos]) == np.argmax(t_output[test_pos]) else 0  
        cos = 1 - spatial.distance.cosine(t_target[test_pos].flatten(), t_output[test_pos].flatten())

        return (acc, acc, acc, cos)

    def test(self, test_sets):
        tests = {}                   
        
        for test_set in test_sets:
            tests[str(test_set)] = ([], [], [], [])

            n = 0            
            for ex in test_set.get_exemplar():
                
                if n >= Settings.n_tests:
                    break
                
                test_input, test_target = ex
                test_output = self._model.predict(np.array([test_input]), batch_size=1)[0]
        
                (task_acc, task_cos, vars_acc, vars_cos) = \
                    self._calc_progress(test_output, test_target)
                    
                
                tests[str(test_set)][0].append(task_acc)
                tests[str(test_set)][1].append(task_cos)
                tests[str(test_set)][2].append(vars_acc)
                tests[str(test_set)][3].append(vars_cos)
                
                n += 1
                
                
            tests[str(test_set)] = (
                    np.mean(tests[str(test_set)][0]), 
                    np.mean(tests[str(test_set)][1]), 
                    np.mean(tests[str(test_set)][2]), 
                    np.mean(tests[str(test_set)][3])
                )
        
        self.__learning_progress.report(tests)        
            
        return tests


    def train(self, train_set, test_sets):

        max_test_acc = 0
        
        n_no_improvement_epochs = 0
        n_epochs = 0
        
        for n_epochs in range(Settings.n_training_epochs):    
            self.__learning_progress.set_trial(n_epochs + 1)
            self._model.fit_generator(
                    generator=train_set.get_batch(),
                    steps_per_epoch = Settings.n_training_batches_per_epoch,
                    epochs = 1, 
                    verbose=0,
                    use_multiprocessing = True,
                    callbacks=[self.__learning_progress]
                    )
            if 0 == ((n_epochs + 1) % Settings.training_report_period):        
                accs = self.test(test_sets)  
                
                rounded_currect_acc = round(accs["test"][2], 3)
                
                if rounded_currect_acc <= max_test_acc:
                    n_no_improvement_epochs += 1
                else:
                    max_test_acc = accs["test"][2]
                    n_no_improvement_epochs = 0
                    
                if n_no_improvement_epochs >= Settings.training_patience:
                    break
                
        return n_epochs,  accs["test"][0], accs["test"][2]

    def _compile(self):
        self._model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(lr=Settings.l_rate))
        

    def _build(self):
        self._model = keras.models.Sequential()
        # self._input = keras.layers.Input(shape=(self._input_dim, ))
        # self._hidden = keras.layers.Input(shape=(self._input_dim, ))
        self._model.add(
                keras.layers.LSTM(
                        units=Settings.n_recurrent_hidden_units, 
                        unroll=True,
                        use_bias=False,
                        return_sequences = True, 
                        input_shape=(Settings.n_roles + 1, self._input_dim)))
        self._model.add(
                keras.layers.TimeDistributed(
                        keras.layers.Dense(
                            units = Settings.n_output_hidden_units,
                            activation  =  "tanh"
                        )
                    )
            )      
        self._model.add(
                keras.layers.TimeDistributed(
                        keras.layers.Dense(
                            units = self._output_dim,
                            activation  =  self._act_function
                        )
                    )
            )

    def __init__(self, name ="BaseModel Model"):
        self.__name = name
        self._act_function = "softmax"
        self._input_dim = Settings.n_fillers + Settings.n_roles + 1
        self._output_dim = Settings.n_fillers        
        self._build()
        self._compile()
        self.__learning_progress = LearningProgressTracker()

class VARSModel(BaseModel):
    def _calc_progress(self, output, target):
        
        #accuracy is calculated only for the last time step, i.e. the test phase
        
        vars_output = expit(output[Settings.n_roles,:Settings.VARS_dim])        
        vars_target = target[Settings.n_roles,:Settings.VARS_dim]
        vars_cos = 1 - spatial.distance.cosine(vars_target, vars_output)
        
        raw_task_output = output[Settings.n_roles, Settings.VARS_dim:]
        stable_task_output = raw_task_output - max(raw_task_output)
        exp_task_output = np.exp(stable_task_output)
        task_output = exp_task_output / np.sum(exp_task_output, None)
        task_target = target[Settings.n_roles, Settings.VARS_dim:]

        #calculate VARS accuracy
        
        #get the indices of the n most active output bits
        #where n is the number of units which are activated (i.e. 1) in the target
        output_max_i = np.argsort(vars_output)[::-1][:np.sum(vars_target)]        
        #sort the indices of the n most active output units
        output_max_i.sort()
        
        #get the indices of the target units which are activated (i.e .1)        
        target_max_i = np.argsort(vars_target)[::-1][:np.sum(vars_target)]
        #sort the indices of the active target units
        target_max_i.sort()

        #output is accurate if the the indices of the most active output units
        #correspond to the indices of the activated target units
        vars_acc = np.array_equal(output_max_i, target_max_i)

        #calculate task accuracy and cosine similarity
        task_acc = 1.0 if np.argmax(task_target) == np.argmax(task_output) else 0
        task_cos = 1 - spatial.distance.cosine(task_target, task_output)
        
        return (task_acc, task_cos, vars_acc, vars_cos)

    def _compile(self):
        def VARS_loss_f(y_true, y_pred):  
#            return 
                    
            VARS_sem_loss =  tf.reduce_mean(
                     tf.nn.sigmoid_cross_entropy_with_logits(
                         logits=y_pred[:, :, :Settings.VARS_sem_space_dim], 
                         labels=y_true[:, :, :Settings.VARS_sem_space_dim]
                 ))

            if Settings.encode_bindings:
                VARS_bind_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=y_pred[:, Settings.n_roles, Settings.VARS_sem_space_dim:Settings.VARS_dim], 
                        labels=y_true[:, Settings.n_roles, Settings.VARS_sem_space_dim:Settings.VARS_dim])
                )
            else:
                VARS_bind_loss = 0
                
            task_loss = tf.reduce_mean(
                     tf.nn.softmax_cross_entropy_with_logits_v2(
                         logits=y_pred[:, :, Settings.VARS_dim:], 
                         labels=y_true[:, :, Settings.VARS_dim:])
                 )
                     
            return VARS_sem_loss + VARS_bind_loss + task_loss
        
        self._model.compile(loss = VARS_loss_f, optimizer = keras.optimizers.Adam(lr = Settings.l_rate))        

    def _build(self):
        self._input_dim = \
            Settings.n_fillers + \
            Settings.n_roles + \
            Settings.n_tokens + 1

        self._output_dim = Settings.VARS_dim + Settings.n_fillers
        self._act_function = "linear"

        BaseModel._build(self)

    def __init__(self, name ="VARS Model"):
        self.__output_dim = Settings.n_tokens * (
            Settings.n_fillers + Settings.n_roles + Settings.n_tokens * Settings.max_arity
        )
        BaseModel.__init__(self, name)
