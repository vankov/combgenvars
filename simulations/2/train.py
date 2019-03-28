from model import VARSModel, VGGModel
from settings import Settings

import glob

simulation_id = 0

while len(glob.glob("results/results*.{}.txt".format(simulation_id))):
    simulation_id += 1

VARS_tasks = [4, 1]

for task in [0]:
    for model_id in [0, 1, 2]:
        for VARS_task in VARS_tasks if model_id == 0 else [0]:
        
            if model_id == 0:               
                #train VARS representations
#                Settings.l_rate = 0.0001
#                Settings.n_training_epochs = 30
                model = VARSModel(use_VARS=1)
            if model_id == 1:               
                #don;t traintrain VARS representations
#                Settings.l_rate = 0.0001
#                Settings.n_training_epochs = 30
                model = VARSModel(use_VARS=0)
                
            if model_id == 2:
                #VGG model
#                Settings.l_rate = 0.0001
#                Settings.n_training_epochs = 30
                model = VGGModel()
        
            Settings.output_file_name = "results/results.{}.{}.{}.{}.txt".format(
                    model_id,
                    task, 
                    VARS_task, 
                    simulation_id
                )

            print("Simulation id: {}".format(simulation_id))
            print("Storing results in {}".format(Settings.output_file_name))
            print("Training model {} on task {} and VARS task {}".format(
                    model_id,
                    task, 
                    VARS_task)
                )
            
            
            model.train(
                    "data/train", 
                    ["data/test"], 
                    targets_file = "targets.{}.{}.txt".format(task, VARS_task)
                )
            
#