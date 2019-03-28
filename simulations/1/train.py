from settings import Settings
from models import BaseModel, VARSModel
import argparse

import sets


parser = argparse.ArgumentParser()
parser.add_argument('-r', type=int, help='role', default=2)
parser.add_argument('-m', type=int, help='model', default=1)                    
parser.add_argument('-b', action='store_true', help='don\'t encode bindings', default=False)       
parser.add_argument('-t', action='store_true', help='don\'t randomize tokens', default=False)     
parser.add_argument('-ro', action='store_true', help='randomize roles order', default=False)
parser.add_argument('-v', action='store_true', help='print information during training', default=False)
args = parser.parse_args()

# train_set = sets.SimpleRNNTrainSet()
# comb_gen_test_set = sets.SimpleRNNTestSet(test_type = "comb_gen")
# anti_test_set = sets.SimpleRNNTestSet(test_type = "anti")

if args.b:
    Settings.encode_bindings = False
if args.t:
    Settings.randomize_tokens = False
if args.ro:
    Settings.fixed_role_order = False

Settings.training_progress_report = args.v
    
Settings.combinatorial_generalization_role_fillers = {}
Settings.combinatorial_generalization_role_fillers[args.r] = [0]

comb_test = sets.CombinatorialGeneralization()
    
if args.m == 0:
    model= BaseModel()
    train_set = sets.BaseModelTrainSet()
    test_set = sets.BaseModelTestSet(test = comb_test)
    
if args.m == 1:
    model = VARSModel()
    train_set = sets.VARSTrainSet()
    test_set = sets.VARSTestSet(test = comb_test)
    


for rep in range(1):
   
    if Settings.training_progress_report:
        print("Training {} on test role {}".format(
                    model.__class__.__name__, 
                    args.r
                ))
    

    n_epochs, task_acc, vars_acc = model.train(train_set, [train_set, test_set])

    if Settings.training_progress_report:
        print("Training completed.")
        print("Task accuracy: {}\nVARS accuracy: {}\n\n".format(task_acc, vars_acc))
    else:
        print("\t".join(
                map(str, [
                    args.m,
                    0 if args.b else 1,
                    args.r,
                    n_epochs, 
                    task_acc, 
                    vars_acc
                ])))