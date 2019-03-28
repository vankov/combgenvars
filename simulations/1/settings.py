import numpy as np

class Settings:
    n_roles = 3
    n_fillers = 10
    
    n_tests = 600

    n_iterations_sets_selection = 1000

    batch_size = 50

    n_training_epochs = 50
    n_training_batches_per_epoch = 5000
    training_report_period = 1
    
    #print training progress
    training_progress_report = True
    
    #learning rate
    l_rate = 0.001

    #proportion of possible examples to include in the training set
    train_set_proportion = 1

    
    #number of epochs to tolerate a non improving validation accuracy
    training_patience = 5

    combinatorial_generalization_role_fillers = {2: [0]}
    spurious_anticorrelation_fillers = [1, 2]

    #number of units in the recurrent layer
    n_recurrent_hidden_units = 100

    #number of units in the hidden layer connecting the recurrent layer
    #and the output
    n_output_hidden_units = 50
    
    #Set to False to make fillers distinct within a trial
    allow_fillier_repetitions = True

    #The probability to a trial twice in the training set if it contains the 
    #target filler
    test_filler_correction = 0
    #Set to False to use arbitrary ordering or roles
    fixed_role_order = [1, 0, 2]

    #VARS settings 
    n_tokens = 3
    max_arity = 2
    
    randomize_tokens = True
    encode_bindings = True

Settings.VARS_sem_space_shape = (Settings.n_tokens, Settings.n_fillers) #semantics space
Settings.VARS_bind_space_shape = (Settings.max_arity, Settings.n_tokens, Settings.n_tokens) #binding(structure) space

Settings.VARS_sem_space_dim = Settings.VARS_sem_space_shape[0] *  Settings.VARS_sem_space_shape[1]
Settings.VARS_bind_space_dim = Settings.VARS_bind_space_shape[0] *  Settings.VARS_bind_space_shape[1] * Settings.VARS_bind_space_shape[2]

Settings.VARS_dim = Settings.VARS_sem_space_dim + Settings.VARS_bind_space_dim

# print(float(Settings.VARS_bind_space_dim) / Settings.VARS_dim)        
Settings.sigma = 0.5#Settings.VARS_bind_space_dim / Settings.VARS_dim
