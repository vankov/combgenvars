from settings import Settings
import numpy as np

class BaseModelEncoder:
    @staticmethod
    def encode(roles, fillers, test_role = None):
        
        if test_role == None:
            test_role = np.random.randint(0, Settings.n_roles)
        
        xs = np.zeros(shape=(
            Settings.n_roles + 1, 
            Settings.n_roles + Settings.n_fillers + 1), 
            dtype=np.int)

        ys = np.zeros(shape=(
            Settings.n_roles + 1,
            Settings.n_fillers), 
            dtype=np.int)

        test_role_filler = -1

        for i in range(Settings.n_roles):
            xs[i][roles[i]] = 1
            xs[i][Settings.n_roles + fillers[i]] = 1
            ys[i][fillers[i]] = 1
            if roles[i] == test_role:
                test_role_filler = fillers[i]

        xs[Settings.n_roles][test_role] = 1
        xs[Settings.n_roles][Settings.n_fillers + Settings.n_roles] = 1
        ys[Settings.n_roles][test_role_filler] = 1

        return (xs, ys)

class VARSEncoder:
    @staticmethod
    def get_space(vector):
        sem = vector[:Settings.VARS_sem_space_dim]
        bind = vector[Settings.VARS_sem_space_dim:]

        return (
            sem.reshape(Settings.VARS_sem_space_shape),
            bind.reshape(Settings.VARS_bind_space_shape)
        )

    @staticmethod
    def get_vector(space):        
        return np.concatenate((space[0].flatten(), space[1].flatten()))

    @staticmethod
    def get_test_role():
        while True:
            for i in range(Settings.n_roles):
                yield i
                
    @staticmethod
    def encode(roles, fillers, test_role = None):
        
        if test_role == None:
            test_role = np.random.randint(0, Settings.n_roles)# next(VARSEncoder.get_test_role())

        xs = np.zeros(shape=(
            Settings.n_roles + 1, 
            Settings.n_roles + Settings.n_fillers + Settings.n_tokens + 1
            ), 
            dtype=np.int)

        ys = np.zeros(shape=(
            Settings.n_roles + 1,
            Settings.VARS_dim + Settings.n_fillers
            ), 
            dtype=np.int)


        if Settings.randomize_tokens:
            tokens = np.random.permutation(Settings.n_tokens)
        else:
            tokens = np.arange(Settings.n_tokens)

        vars = (
                np.zeros(shape=Settings.VARS_sem_space_shape),
                np.zeros(shape=Settings.VARS_bind_space_shape),
        )

        role_i = -1
        ro1_i = -1
        ro2_i = -1
        
        for i in range(Settings.n_roles):
            #set role
            xs[i][roles[i]] = 1
            #set filler
            xs[i][Settings.n_roles + fillers[i]] = 1
            #set token
            xs[i][Settings.n_roles + Settings.n_fillers + tokens[i]] = 1

            #set VARS filler            
            vars[0][tokens[i]][fillers[i]] = 1

            if (roles[i] == test_role):
                test_filler = fillers[i]

            if Settings.encode_bindings:        
                if (roles[i] == 0):
                    role_i = i
                    if ro1_i > -1:
                        vars[1][0][tokens[i]][tokens[ro1_i]] = 1
                    if ro2_i > -1:
                        vars[1][1][tokens[i]][tokens[ro2_i]] = 1                    
                if (roles[i] == 1):
                    ro1_i = i
                    if role_i > -1:
                        vars[1][0][tokens[role_i]][tokens[i]] = 1
                if (roles[i] == 2):
                    ro2_i = i
                    if role_i > -1:
                        vars[1][1][tokens[role_i]][tokens[i]] = 1
                        
                
            #set VARS target
            ys[i][:Settings.VARS_dim] = VARSEncoder.get_vector(vars)                
                
            #set task target filler
            ys[i][Settings.VARS_dim + fillers[i]] = 1


        if Settings.encode_bindings:
            assert(role_i > -1)
            assert(ro1_i > -1)
            assert(ro2_i > -1)

        # print(tokens)
#        print(roles)
#        print(fillers)
        xs[Settings.n_roles][test_role] = 1        
        xs[Settings.n_roles][Settings.n_roles + Settings.n_fillers + Settings.n_tokens] = 1        
        ys[Settings.n_roles][:Settings.VARS_dim] = VARSEncoder.get_vector(vars)
        ys[Settings.n_roles][Settings.VARS_dim + test_filler] = 1



#        # print(xs)
#        print(VARSEncoder.get_space(ys[0][:Settings.VARS_dim]))        
#        print(VARSEncoder.get_space(ys[1][:Settings.VARS_dim]))        
#        print(VARSEncoder.get_space(ys[2][:Settings.VARS_dim]))        
#        print(VARSEncoder.get_space(ys[Settings.n_roles][:Settings.VARS_dim]))
#        exit(0)
        #     print(xs[i])
        #     print(vars[0])
        #     print(vars[1])

        # exit(0)
        return (xs, ys)