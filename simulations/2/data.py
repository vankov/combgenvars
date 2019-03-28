from __future__ import print_function

import numpy as np

import os
import shutil
from PIL import Image, ImageDraw, ImageColor
from itertools import permutations, product

from settings import Settings

class Exemple:

    #static object counter
    _id_counter = 0

    # cache, static dictionary with all objects
    _objects = {}
    
    def get_id(self):
        return self.__id
    
    def save(self, data_dir, tasks = None, VARS_tasks = None):
                
        with open("{}/images/{}.png".format(data_dir, self.get_id()), "w") as img_file_h:
            self.__image.save(img_file_h)            
            
        if tasks is None:
            tasks = [Settings.task]
            
        if VARS_tasks is None:
            VARS_tasks = [Settings.VARS_task]
            
        for VARS_task in VARS_tasks:
            
            (semantics, bindings) = self.get_VARS_target(VARS_task)    
            
            for task in tasks:            
                
                target = np.concatenate(
                            (
                                [int(self.get_id())], 
                                semantics.flatten(), 
                                bindings.flatten(),
                                self.get_task_target(task)
                            )
                        )
                targets_file_path = "{}/targets.{}.{}.txt".format(
                            data_dir,
                            task,
                            VARS_task
                        )
                with open(targets_file_path, "a") as target_file_h:
                    target_file_h.write("\t".join(map(str, target)))
                    target_file_h.write("\n")
                        
    def __get_tokens(self):
        return self.__tokens
    
    def get_VARS_target(self, VARS_task = None):
        
        if VARS_task is None:
            VARS_task = Settings.VARS_task
          
            
        if ((VARS_task > 0 and Settings.n_tokens < 3)
                or (VARS_task > 1 and Settings.n_tokens < 4)):
            
            raise "Not enough tokens."
        
        semantics = np.zeros(shape = Settings.VARS_sem_space_shape, dtype=np.int)
        bindings = np.zeros(shape = Settings.VARS_bind_space_shape, dtype=np.int)
        
        if VARS_task > 0:
            #encode objects
            
            for i in range(3):
                semantics[self.__tokens[i]][i] = 1
#                semantics[self.__tokens[i]][self.__colors[i]] = 1
#                semantics[self.__tokens[i]][Settings.n_colors + self.__shapes[i]] = 1

                
        same_color_rel = self.get_same_color_rel()
        same_shape_rel = self.get_same_shape_rel()

        if same_color_rel:
            same_rel = same_color_rel
        else:
            same_rel = same_shape_rel

        if VARS_task == 2:
            #encode odd(X)
                
            if Settings.max_arity < 1:
                raise "Insufficient maximum arity to represent odd(X)"
                
            semantics[self.__tokens[3]][Settings.VARS_sem_dim - 1] = 1
            bindings[0][self.__tokens[3]][self.__tokens[same_rel[2]]] = 1
            
        if VARS_task == 3:
            #encode same((X, Y))
                
            if Settings.max_arity < 1:
                raise "Insufficient maximum arity to represent same((X, Y))"
                
            semantics[self.__tokens[3]][Settings.VARS_sem_dim - 1] = 1
            bindings[0][self.__tokens[3]][self.__tokens[same_rel[0]]] = 1
            bindings[0][self.__tokens[3]][self.__tokens[same_rel[1]]] = 1

        if VARS_task == 4:
            #encode different-from(X, (Y, Z))
                        
            if Settings.max_arity < 2:
                raise "Insufficient maximum arity to represent different-from(X, (Y, Z))"
                
            semantics[self.__tokens[3]][Settings.VARS_sem_dim - 1] = 1
            bindings[0][self.__tokens[3]][self.__tokens[same_rel[2]]] = 1
            bindings[1][self.__tokens[3]][self.__tokens[same_rel[0]]] = 1
            bindings[1][self.__tokens[3]][self.__tokens[same_rel[1]]] = 1
        
        return (semantics, bindings)
    
    def get_odd_object(self):
        same_col_rel = self.get_same_color_rel()
        if same_col_rel:
            return same_col_rel[2]
        else:
            same_shape_rel = self.get_same_shape_rel()
            return same_shape_rel[2]

    def get_same_color_rel(self):
        
        if self.__colors[0] == self.__colors[1]:
            return (0, 1, 2)

        if self.__colors[0] == self.__colors[2]:
            return (0, 2, 1)

        if self.__colors[1] == self.__colors[2]:
            return (1, 2, 0)
        
        return None

    def get_same_shape_rel(self):
        
        if self.__shapes[0] == self.__shapes[1]:
            return (0, 1, 2)

        if self.__shapes[0] == self.__shapes[2]:
            return (0, 2, 1)

        if self.__shapes[1] == self.__shapes[2]:
            return (1, 2, 0)
        
        return None
        
        
    def get_task_target(self, task = None):
        
        target = np.zeros(shape=Settings.task_dim, dtype = np.int)       
        
        if task is None:
            task = Settings.task
            
        if task == 0:
            #position
            target[self.get_odd_object()] = 1
            
        if task == 1:
            #color
            target[self.__colors[self.get_odd_object()]] = 1            
            
        if task == 2:
            #shape
            target[self.__shapes[self.get_odd_object()]] = 1            
            
        return target
    
    def _prepare_objects(self):     
        if not self._objects:
            for color in range(Settings.n_colors):
                for shape in range(Settings.n_shapes):
                     im = Image.open("shapes/{}.png".format(shape))   
                     pixels = im.load()
                     for y in xrange(im.size[1]): 
                         for x in xrange(im.size[0]):
                             if pixels[x,y][0] == 255:
                                 pixels[x,y] = (0, 0, 0)
                             else:
                                 pixels[x,y] = ImageColor.getrgb(Settings.colors[color])
                     im.thumbnail((Settings.shape_w, Settings.shape_h), Image.ANTIALIAS)
                     
                     Exemple._objects["{}-{}".format(shape, color)] = im
                 
    def _draw_object(self, position, color, shape):
                             
         self.__image.paste(
                     Exemple._objects["{}-{}".format(shape, color)],
                     box = (
                        Settings.image_w / 2 - Settings.shape_w / 2, 
                        position * Settings.image_h / 3 + (Settings.image_h / 3 - Settings.shape_h) / 2,
                        Settings.image_w / 2 + Settings.shape_w / 2, 
                        position * Settings.image_h / 3 
                            + (Settings.image_h / 3 - Settings.shape_h) / 2
                            + Settings.shape_h,
                     )
                 )

                    
    def draw(self):
        
        for i in range(len(self.__colors)):
            self._draw_object(
                    i, 
                    self.__colors[i],
                    self.__shapes[i]
                )
            
    def has_replacement(self):
        return False
    
    def get_replacement(self):
        raise("{} has no replacement".format(self.__class__.__name__))
                            
    def get_shapes(self):
        return self.__shapes
    
    def get_colors(self):
        return self.__colors
    
    def __init__(self, colors, shapes):        
        Exemple._id_counter += 1
        self.__id = Exemple._id_counter
        
        self.__colors = colors
        self.__shapes = shapes
        
        self.__image = Image.new('RGB', (Settings.image_w, Settings.image_h), color='black')
        self.__canvas = ImageDraw.Draw(self.__image)
        self._prepare_objects()

        self.__tokens = np.arange(Settings.n_tokens)


    @staticmethod
    def generate(example_class):
	
        for colors in product(np.arange(Settings.n_colors), repeat = 3):
            for shapes in product(np.arange(Settings.n_shapes), repeat = 3):
                if (
                        (
                            (colors[0] == colors[1] and colors[1] == colors[2])
                            or (shapes[0] == shapes[1] and shapes[1] == shapes[2])
                        ) 
                        or (
                            (colors[0] <> colors[1])
                            and (colors[1] <> colors[2])
                            and (colors[0] <> colors[2])
                            and (shapes[0] <> shapes[1]) 
                            and (shapes[1] <> shapes[2])
                            and (shapes[0] <> shapes[2])
                        )
                    ):
                        continue

                example = example_class(
                        colors = colors,
                        shapes = shapes
                    )
                if example.get_same_color_rel() and example.get_same_shape_rel():
                    continue
                
                if example_class.check(
                        colors = colors,
                        shapes = shapes
                    ):
                        if np.random.rand() <= example_class.get_frequency():
                            yield example
                else:
                    if example.has_replacement():
                        yield example.get_replacement()


class TrainExemple(Exemple):
    
    def has_replacement(self):
        return False
    
    def get_replacement(self):
        odd_shape_i = self.get_odd_object()
        shapes = [
                    self.get_shapes()[odd_shape_i]
                ]
        colors = [
                    self.get_colors()[odd_shape_i]
                ]
        
        while True:
            color2 = self.get_colors()[np.random.randint(3)]
            if color2 <> colors[0]:
                break
            
        colors.append(color2)
        colors.append(color2)
        
        while True:
            shape2 = self.get_shapes()[np.random.randint(3)]
            if  shape2 <> shapes[0]:
                break
        shapes.append(shape2)
        
        while True:
            shape3 = np.random.randint(Settings.n_shapes)
            if not(shape3 in shapes):
                break
        shapes.append(shape3)
                
        order = list(next(permutations([0, 1, 2])))
        
        return Exemple(
                colors = np.array(colors)[order], 
                shapes = np.array(shapes)[order]
            )
    
    @staticmethod
    def get_frequency():
        return Settings.freq_training_examples
    
    @staticmethod
    def check(colors, shapes):
        
        return not(CombinatorialGeneralizationTestExample.check(colors, shapes))
                    
class CombinatorialGeneralizationTestExample(Exemple):
    def has_replacement(self):
        return False;
    
    @staticmethod
    def get_frequency():
        return 1
    
    @staticmethod
    def check(colors, shapes):

        return (
                    (
                            ((None == Settings.test_shape) or shapes[0] == Settings.test_shape)
                            and
                            ((None == Settings.test_color) or colors[0] == Settings.test_color)
                            and
                            (
                                    (shapes[1] == shapes[2])
                                    or
                                    (colors[1] == colors[2])
                            )
                    )
                    or
                    (
                            ((None == Settings.test_shape) or shapes[1] == Settings.test_shape)
                            and
                            ((None == Settings.test_color) or colors[1] == Settings.test_color)
                            and
                            (
                                    (shapes[0] == shapes[2])
                                    or
                                    (colors[0] == colors[2])
                            )
                    )                
                    or
                    (
                            ((None == Settings.test_shape) or shapes[2] == Settings.test_shape)
                            and
                            ((None == Settings.test_color) or colors[2] == Settings.test_color)
                            and
                            (
                                    (shapes[0] == shapes[1])
                                    or
                                    (colors[0] == colors[1])
                            )                            
                    )                  
                ) 
                
class DataSet:
    @staticmethod
    def generate(data_dir, set_class, tasks = None, VARS_tasks = None):
        
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            
        os.makedirs(data_dir)
        os.makedirs("{}/images".format(data_dir))
        
        for st in Exemple.generate(set_class):
            st.draw()
            target_dir = "{}".format(data_dir)
            st.save(
                        target_dir,
                        tasks = tasks,
                        VARS_tasks = VARS_tasks,
                    )
            
class TrainSet:
    @staticmethod
    def generate(train_data_dir, tasks = None, VARS_tasks = None):
        DataSet.generate(
                train_data_dir, 
                TrainExemple, 
                tasks = tasks, 
                VARS_tasks = VARS_tasks
            )
            
class CombinatorialGeneralizationTestSet:
    @staticmethod
    def generate(test_data_dir, tasks = None, VARS_tasks = None):            
        DataSet.generate(
                test_data_dir, 
                CombinatorialGeneralizationTestExample, 
                tasks = tasks, 
                VARS_tasks = VARS_tasks
            )

