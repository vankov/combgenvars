class Settings:
    n_rnd_examples = 1 
    
    image_w = 48
    image_h = 48
        
    shape_w = 12
    shape_h = 12
    
            
    n_colors = 6
    n_shapes = 6
    
    #properties of the test odd object
    #set to None in order not to use
    #For exameple, setting test_color = None defines the test object
    #by shape only
    
    test_color = 0
    test_shape = None
    
    #which property of the odd object to output 
    #0 - position
    #1 - color
    #2 - shape
    task = 0
    #the numer of output units for the particular task
    task_dim = 3
    
    #symbols to encode in the vars target
    #0 - None
    #1 - objects only
    #2 - objects and odd(X)
    #3 - objects and same((X, Y))
    #4 - objects and different-from(X, (Y, Z))
    #odd(X)
    VARS_task = 4
    
    n_iterations_sets_selection = 10000

    batch_size = 100
    
    n_training_epochs = 50
    n_training_batches_per_epoch = 500
    training_report_period = 1
    
    n_test_batches = 1
    
    l_rate = 0.0001
    
    l_rate_update_freq = 100
    l_rate_decay = 1
    
    #turn on to save the best performing state of the model during training
    save_model_best_state = False
    #turn on to save the current state of the model during training
    save_model_current_state = False

    #turn on to print VARS error and exit script, used for debugging
    print_error_and_stop = False
    
    #VARS settings 
    n_tokens = 4
    max_arity = 2
    
    #randomize_tokens for each training example
    #not needed if it is expected that each object can appear
    #at any position and we always have a fixed number of symbols
    #to represent
    randomize_tokens = True
    
    #probability to include a examples in the training set
    freq_training_examples = 1
    
    #simulation output file name
    output_file_name = None

Settings.VARS_sem_dim = 3 + 1#Settings.n_colors + Settings.n_shapes + 1

Settings.VARS_sem_space_shape = (Settings.n_tokens, Settings.VARS_sem_dim) #semantics space
Settings.VARS_bind_space_shape = (Settings.max_arity, Settings.n_tokens, Settings.n_tokens) #binding(structure) space

Settings.VARS_sem_space_dim = Settings.VARS_sem_space_shape[0] *  Settings.VARS_sem_space_shape[1]
Settings.VARS_bind_space_dim = Settings.VARS_bind_space_shape[0] *  Settings.VARS_bind_space_shape[1] * Settings.VARS_bind_space_shape[2]

Settings.VARS_dim = Settings.VARS_sem_space_dim + Settings.VARS_bind_space_dim

#relative weight of semantics and bindings.
#currently not used by in the loss function, but usefull for assessing 
#the cosine similarity of the targets and outputs
Settings.sigma = 0.5

Settings.colors = [
            "green",
            "red", 
            "blue",
            
            "white",
            "cyan",            
            "yellow", 
            
            "purple",
            "navy", 
            "lime",
            "violet",            
            "magenta",
            
            ]

#set task dim
task_dim = max(3, Settings.n_colors, Settings.n_shapes)
if Settings.task == 0:
    #position
    task_dim = 3
    
if Settings.task == 1:
    #color
    task_dim = Settings.n_colors
    
if Settings.task == 2:
    #shape
    task_dim = Settings.n_shapes