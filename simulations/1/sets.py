import numpy as np
from settings import Settings
from abc import abstractmethod
import itertools
from encoders import VARSEncoder, BaseModelEncoder
from scipy.spatial.distance import cosine
import pickle

class TestCondition:
    @abstractmethod
    def check_exemplar(self, roles, fillers):
        pass

    @abstractmethod
    def name(self):
        pass

    def get_test_role(self):
        None
    
class CombinatorialGeneralization(TestCondition):
    
    def get_test_role(self):        
        return np.random.choice(self.__test_roles, size = 1)[0]
        
    def check_exemplar(self, roles, fillers):

        for i in range(Settings.n_roles):
            for role in Settings.combinatorial_generalization_role_fillers:
                if roles[i] == role and fillers[i] in Settings.combinatorial_generalization_role_fillers[role]:
                    return role

        return False

    def __str__(self):
        return "comb test"

    def __init__(self):
        self.__test_roles = []
        for role in Settings.combinatorial_generalization_role_fillers:
            self.__test_roles.append(role)

class SpuriousAnticorrelation(TestCondition):
    def check_exemplar(self, roles, fillers):
        n_fillers_in_exception = 0
        for i in range(Settings.n_roles):
            if fillers[i] in Settings.spurious_anticorrelation_fillers:
                n_fillers_in_exception += 1
                if n_fillers_in_exception >= len(Settings.spurious_anticorrelation_fillers):
                    return roles[i]
        return False

    def __str__(self):
        return "anti test"

class Set:

    def __str__(self):
        return "abstract set"
    
    def get_fillers_generator(self):
        
        if Settings.allow_fillier_repetitions:
            return itertools.product(
                        np.arange(Settings.n_fillers),
                        np.arange(Settings.n_fillers),
                        np.arange(Settings.n_fillers)
                        )                    
        else:
            return itertools.permutations(
                        np.arange(Settings.n_fillers)
                    )
    
        
    def get_roles_generator(self):
        
        if Settings.fixed_role_order is False:        
            return itertools.permutations(np.arange(Settings.n_roles), r = 3)
        else:
            return [Settings.fixed_role_order]    

    
    def get_batch(self):
        while True:
            batch = ([], [])
            for ex in self.get_exemplar():                
                batch[0].append(ex[0])
                batch[1].append(ex[1])
                if len(batch[0]) == Settings.batch_size:
                    yield (np.array(batch[0]), np.array(batch[1]))
                    batch = ([], [])
        
    def get_exemplar(self):
        raise("Not defined")

    def __init__(self, encoder):
        self._encoder = encoder


class TrainSet(Set):
    
    def __str__(self):
        return "abstract train set"
    
    def get_exemplar(self):
        
        np.random.shuffle(self.__data)
        for ex in self.__data:
            yield self._encoder.encode(ex[0], ex[1]) 
            
                
            
            
    def __init__(self, encoder):
        test = CombinatorialGeneralization()
        self.__data = []
#
        for roles in self.get_roles_generator():
            for fillers in self.get_fillers_generator():
                if test.check_exemplar(roles, fillers) is False:
                    self.__data.append((roles, fillers))
                    if 0 in fillers:
                        if np.random.rand() < Settings.test_filler_correction:
                            self.__data.append((roles, fillers))

        np.random.shuffle(self.__data)         
        full_train_set_size = len(self.__data)
        if Settings.train_set_proportion > 0:
            self.__data = self.__data[
                        :int(len(self.__data) * Settings.train_set_proportion)
                    ]
        if Settings.training_progress_report:
            print("Training {} out of {} examples.".format(
                        len(self.__data),
                        full_train_set_size
                    ))
            
        Set.__init__(self, encoder)
        
class TestSet(Set):
   
    def __str__(self):
        return "abstract test set"
    
    def get_exemplar(self):
        np.random.shuffle(self.__data)
        for ex in self.__data:
            yield self._encoder.encode(ex[0], ex[1], self.__test_role) 


    def __init__(self, test, encoder):
        self._test = test      
        self.__test_role = test.get_test_role()
        self.__data = []
            
        for roles in self.get_roles_generator():
            for fillers in self.get_fillers_generator():
                if not self._test.check_exemplar(roles, fillers) is False:
                   self.__data.append((roles, fillers))
                   
        if Settings.training_progress_report:
            print("Test set consists of {} examples.".format(len(self.__data)))
        Set.__init__(self, encoder)


class BaseModelTrainSet(TrainSet):
    def __str__(self):
        return "train"
    
    def __init__(self):
        TrainSet.__init__(self, encoder = BaseModelEncoder())

class BaseModelTestSet(TestSet):
    def __str__(self):
        return "test"
    
    def __init__(self, test):
        TestSet.__init__(self, test, BaseModelEncoder())


class VARSTrainSet(TrainSet):
    def __str__(self):
        return "train"
            
    def __init__(self):
        TrainSet.__init__(self, encoder = VARSEncoder())

class VARSTestSet(TestSet):
    def __str__(self):
        return "test"
                
    def __init__(self, test):
        TestSet.__init__(self, test, VARSEncoder())

