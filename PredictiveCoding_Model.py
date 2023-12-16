import random
import copy
import numpy as np
import string
import matplotlib.pyplot as plt
import bz2
import _pickle as cPickle
import os


random.seed(1)



def wordlist_to_orth(wordlist):
    # takes in list of K words, outputs a 104 x K "spelling" array
    alphabet = string.ascii_lowercase
    wordids = np.array([np.array([alphabet.index(L) for L in word]) for word in wordlist])
    onehots = np.zeros((len(wordlist), len(wordlist[0])*len(alphabet)))
    for i in range(len(wordlist)):
        indices = np.add(wordids[i],np.array([0,1,2,3])*26)
        onehots[i,:][indices] = 1
    return onehots.T

def wordlist_to_ctx(word_stimlist, lexicon, cloze = 0.99, preact_resource = 2.0):
    # takes in a list of K words, assigns a high probability (p = cloze) to each of them in separate trials
    # preact_resource is the value that the pseudoprobability distribution sums to
    lexicon_indices = np.array([lexicon.index(w) for w in word_stimlist])
    low_prob = np.multiply(np.ones((len(lexicon),len(word_stimlist))), preact_resource / (len(lexicon) - 1))* (1 - cloze)
    low_prob[lexicon_indices, np.arange(len(lexicon_indices))] = preact_resource*cloze
    return low_prob

def orth_to_wordlist(onehots):
    # change an N x 104 matrix of one-hot encodings to a list of words
    alphabet = string.ascii_lowercase
    NumWords = onehots.shape[0]
    LettersPerWord = int(onehots.shape[1]/len(alphabet))
    matrix_perword = onehots.reshape((NumWords,LettersPerWord,26))
    wordlist = []
    for w in range(NumWords):
        letterlist = [alphabet[i] for i in np.where(matrix_perword[w][:])[-1]]
        wordlist.append(''.join(letterlist))
    return wordlist


def get_summary(model):
    '''
    takes in a statespace at a given iteration, and records state unit and error unit information at the lexical and semantic levels 
    Arguments:
        statespace: the Simulation.statespace object where state unit and prediction error information exists

    '''

    summary_dict = {}
    
    # DECIDE WHAT INFORMATION TO EXTRACT HERE
    # kinds = ['state', 'reconstruction', 'preactivation', 'prediction_error']
    summary_dict['all_lex_states'] =  model.statespace['lex'][0]
    summary_dict['all_lex_PE'] =  model.statespace['lex'][3]
    summary_dict['all_sem_states'] =  model.statespace['sem'][0]
    summary_dict['all_sem_PE'] =  model.statespace['sem'][3]

    return summary_dict


class Simulation:
    def __init__(self, **kwargs):
    
        default_args = {
            "sim_input":[],# list of words
            "clamp_iterations" : 1, # how many iterations to run the input
            "blanks_before_clamp":0 , # number of blank trials before clamping input
            "BU_TD_mode": "bottom_up", # inputs can be "bottom_up" or "top_down"
            "cloze" : None, # this only applies for top-down inputs
            "summary_funcs": {} ,# dict specifying functions that extract summary info from the model statespace (e.g., PEtotal) at each iteration
            "prevSim" : {}, # if this is a simulation running immediately after another simulation
             "EPSILON1" : 1e-2, # hyperparameter for elementwise division
             "EPSILON2" : 1e-4,  # hyperparameter for elementwise multiplication
             "preact_resource" : 2,
             "sim_filename": False,
             "iterations_this_run" : 0,
             "simulation_data": {},
             "individual_items": False,
             "input_noise": (0,0),
             "weight_noise": (0,0)
             
        }
        
        if os.path.exists(f"./data/{kwargs.get('sim_filename')}" + '.pbz2'):
            data = bz2.BZ2File(f"./data/{kwargs.get('sim_filename')}" + '.pbz2', 'rb')
            data = cPickle.load(data)
            self.__dict__.update(data)
        else:
            for (prop, default) in default_args.items():
                setattr(self, prop, kwargs.get(prop, default))
            if self.prevSim.get('sim_filename',None) != None:
                self.load_prevSimulation()
            else:
                self.define_weights()
                self.define_statespace()            
            assert self.BU_TD_mode in ["bottom_up", "top_down"] # only the orthographic or contextual layers may have inputs defined for them
            if self.BU_TD_mode == "bottom_up":
                self.input = wordlist_to_orth(self.sim_input)
            else:
                self.input = wordlist_to_ctx(self.sim_input, self.lexicon.words, cloze = self.cloze, preact_resource = self.preact_resource)
            self.run_simulation()
    def load_prevSimulation(self):
        '''
        load a copy of the previous simulation, update the statespace, weights, lexicon and current_iteration of the model;
        then delete the prevSimulation copy. Keeping it around consumes too much memory.
        '''
        data = bz2.BZ2File(f'./data/{self.prevSim["sim_filename"]}' + '.pbz2', 'rb')
        data = cPickle.load(data)
        data_to_load = ['weights','statespace', 'current_iteration','lexicon','simulation_data']
        # load weights
        for key in data_to_load:
            setattr(self, key, copy.deepcopy(data[key]))
        
        data.clear()

    def define_statespace(self):

        self.statespace = {'orth': np.ones((4,104,len(self.sim_input)))/26, # first four dims are st/r/preact/PE
                            'lex': np.ones((4,self.lexicon.size,len(self.sim_input)))/self.lexicon.size,
                            'sem': np.ones((4,self.lexicon.semfeatmatrix.shape[0], len(self.sim_input)))/self.lexicon.semfeatmatrix.shape[0],
                            'ctx': np.ones((4,self.lexicon.size,len(self.sim_input)))/self.lexicon.size}
        self.statespace['ctx'][1:,:,:] = np.nan # contextual level doesn't have reconstructions, preactivations or prediction errors 
        self.statespace['lex'][2,:,:] = np.zeros((self.lexicon.size,len(self.sim_input)))
        self.statespace['sem'][2,:,:] = np.zeros((self.lexicon.semfeatmatrix.shape[0], len(self.sim_input)))
        self.current_iteration = 0

    def define_weights(self):

        self.weights = {}
        self.lexicon = Lexicon()
        W1 = wordlist_to_orth(self.lexicon.words).T
        # define orthographic-to-lexical mapping
        self.weights['divide_wt_O_to_L'] = np.dot(np.block([W1, np.eye(self.lexicon.size)]), np.ones((self.lexicon.size + 104,1)))
        self.weights['O_to_L'] = np.divide(W1, self.weights['divide_wt_O_to_L'])
        self.weights['L_to_O'] = self.weights['O_to_L'].T
        # define lexical-to-semantic, and semantic-to-contextual mapping
        self.weights['L_to_S'] = self.lexicon.semfeatmatrix
        self.weights['S_to_C'] = self.lexicon.semfeatmatrix.T

        self.weights['divide_wt_L_to_S'] = np.dot(np.block([self.weights['L_to_S'],  np.eye(self.lexicon.semfeatmatrix.shape[0])]), np.ones((self.lexicon.size +self.lexicon.semfeatmatrix.shape[0],1)))
        self.weights['L_to_S'] = np.divide(self.weights['L_to_S'], self.weights['divide_wt_L_to_S'])
        self.weights['S_to_L'] = self.weights['L_to_S'].T

        self.weights['divide_wt_S_to_C'] = np.dot(self.weights['S_to_C'],np.ones((self.lexicon.semfeatmatrix.shape[0],1)))
        self.weights['S_to_C'] = np.divide(self.weights['S_to_C'], self.weights['divide_wt_S_to_C'])
        self.weights['C_to_S'] = self.weights['S_to_C'].T

        # add frequency bias to non-zero feedback weights
        self.weights['L_to_O'] = np.multiply(self.weights['L_to_O'] + self.lexicon.frequency, self.lexicon.orthmatrix > 0) 
        self.weights['S_to_L']  = np.multiply(self.weights['S_to_L'] + self.lexicon.frequency.T, self.lexicon.semfeatmatrix.T > 0)
        self.weights['C_to_S'] = np.multiply(self.weights['C_to_S'] + self.lexicon.frequency, self.lexicon.semfeatmatrix > 0)
        # OPTIONAL: add weight noise to feedback weights
        z_1 = np.abs(np.random.normal(loc = self.weight_noise[0], scale = self.weight_noise[1], size = self.weights['L_to_O'].shape))
        z_2 = np.abs(np.random.normal(loc = self.weight_noise[0], scale = self.weight_noise[1], size = self.weights['S_to_L'].shape))
        z_3 = np.abs(np.random.normal(loc = self.weight_noise[0], scale = self.weight_noise[1], size = self.weights['C_to_S'].shape))
        z_1[z_1 < 0] = 0
        z_2[z_2 < 0] = 0
        z_3[z_3 < 0] = 0
        self.weights['L_to_O'] += z_1
        self.weights['S_to_L'] += z_2
        self.weights['C_to_S'] += z_3
    def run_one_iteration(self):
        def eps_div(x,y):
            return np.divide(x, np.clip(y, a_min = self.EPSILON1, a_max = np.inf))
        def eps_mul(x,y):
            return np.multiply(np.clip(x, a_min = self.EPSILON2, a_max = np.inf), y)
        def get_wt(x,y):
            return ''.join(['OLSC'[x],'_to_','OLSC'[y]])

        levels = ['orth','lex','sem','ctx']
        kinds = ['state', 'reconstruction', 'preactivation', 'prediction_error']

        # compute states, prediction errors and preactivations in that order for each level of representation (orth -> lex -> sem) 
        # except ctx, which doesn't have rcn, err or preact associated with it
        for lvl in range(3):
            # for the orth level in bottom-up mode, set the input either as blanks or self.input depending on number of blank iterations
            # for the orth level in top-down mode, just update it with preact
            if levels[lvl] == 'orth':
                if self.iterations_this_run < self.blanks_before_clamp:
                    self.statespace['orth'][0] = np.zeros((self.statespace['orth'][0].shape))
                else:
                    if self.BU_TD_mode == "bottom_up":
                        self.statespace['orth'][0] = self.input + np.random.normal(loc = self.input_noise[0], scale = self.input_noise[1], size = self.input.shape)# st <- input
                    else:    
                        self.statespace['orth'][0] = eps_mul(self.statespace['orth'][0], self.statespace['orth'][2])
                self.statespace['orth'][3] = eps_div(self.statespace['orth'][0] , self.statespace['orth'][1]) # err <- st ./ max(rcn, eps1)
                self.statespace['orth'][2] =  eps_div(self.statespace['orth'][1], self.statespace['orth'][0]) # preact <- rcn ./ max(st,eps1); won't be needed if mode is bottom up
            else:
                update_term = np.dot(self.weights[get_wt(lvl-1,lvl)], self.statespace[levels[lvl-1]][3]) + self.statespace[levels[lvl]][2] # update <- W*err + preact
                self.statespace[levels[lvl]][0] = eps_mul(self.statespace[levels[lvl]][0], update_term) # st <- st .* update
                div_term = np.divide(np.eye(self.lexicon.size),self.weights['divide_wt_O_to_L']) if lvl == 1 else np.divide(np.eye(self.lexicon.semfeatmatrix.shape[0]),self.weights['divide_wt_L_to_S'])
                self.statespace[levels[lvl]][3] = eps_div(self.statespace[levels[lvl]][0], self.statespace[levels[lvl]][1]) # err <- st ./ max(rcn, eps1)
                self.statespace[levels[lvl]][2] =  np.dot(div_term, eps_div(self.statespace[levels[lvl]][1], self.statespace[levels[lvl]][0])) # preact <- rcn ./ max(st,eps1)
        self.statespace['ctx'][0]  = eps_mul(self.statespace['ctx'][0], np.dot(self.weights['S_to_C'], self.statespace['sem'][3])) \
            if self.BU_TD_mode == "bottom_up" else self.input
        # compute all reconstructions
        for lvl in range(3):
            self.statespace[levels[lvl]][1] = np.dot(self.weights[get_wt(lvl+1, lvl)], self.statespace[levels[lvl+1]][0])        
        
        self.current_iteration += 1
        self.iterations_this_run += 1
        print(f'Just finished iteration {self.current_iteration}')
    def extract_info(self):
        # take the statespace at each iteration and extract an arbitrary value from it
        if self.simulation_data == {}:
            self.simulation_data = copy.deepcopy(get_summary(self)) 
        else:
            for key,val in get_summary(self).items():
                self.simulation_data[key] = np.dstack((self.simulation_data[key], val))
    def run_simulation(self):
        if self.prevSim == {}:
            self.extract_info() # if there was a prevSim, this info has already been extracted. This line basically avoids extracting the same info twice
        for iteration in range(self.blanks_before_clamp + self.clamp_iterations):
            self.run_one_iteration()
            self.extract_info()
        if self.sim_filename != None:
            # Create the data folder if it does not exist
            if not os.path.exists('./data/'):
                os.makedirs('./data/')
            with bz2.BZ2File(f'./data/{self.sim_filename}.pbz2', 'w') as f:
                save_dict = {key: val for key,val in self.__dict__.items()} 
                cPickle.dump(save_dict, f)

class Lexicon:
    def __init__(self, **kwargs):
        # import the data
        with open(r'./helper_txt_files/1579words_words.txt') as f:
            lexicon_words = f.read()
            lexicon_words = lexicon_words.split('\n')
        with open(r'./helper_txt_files/1579words_ONsize.txt') as f:
            ONvals = f.read()
            ONvals = ONvals.split('\n')
            ONsize = np.array([float(i) for i in ONvals])
            ONsize = np.array([ONsize]) # this is a transposable 1 x 1579 row  
        with open(r'./helper_txt_files/1579words_freq_values.txt') as f:
            freqvals = f.read()
            freqvals = freqvals.split('\n')
            freq = np.array([float(i) for i in freqvals])
            freq = np.array([freq]) # this is a transposable 1 x 1579 row                
        with open(r'./helper_txt_files/1579words_conc_values.txt') as f:
            concvals = f.read()
            concvals = concvals.split('\n')
            conc = np.array([int(i) for i in concvals])
            conc = np.block([conc, np.zeros(len(lexicon_words) - len(conc),dtype = int)])

        default_args = {
            "words":lexicon_words,# list of words
            "size": len(lexicon_words),
             "ONsize" : ONsize,
            "frequency" : freq, 
            "concreteness" : conc
        }
        for (prop, default) in default_args.items():
            setattr(self, prop, kwargs.get(prop, default))

        def repelem(x,y):
            return np.repeat(np.eye(x),y, axis = 0)
        
        # create the semantic feature matrix with arbitrary features.
        shared_feats_block = np.block([repelem(2**i,2**(9-i)) for i in range(9,0,-1)]).T # number of shared features is 9
        conc_feats = repelem(256,9)
        concrete_block = np.zeros((conc_feats.shape[0],512))
        for count, col in enumerate(np.nonzero(self.concreteness)[0]):
            concrete_block[:,col] = conc_feats[:,count] 
        shared_and_conc_block = np.vstack([shared_feats_block, concrete_block])
        num_filler_items = self.size - 512
        pad_with_zero = np.block([shared_and_conc_block, np.zeros((shared_and_conc_block.shape[0], num_filler_items))])
        filler_feats = np.block([np.zeros((num_filler_items*9,512)), repelem(num_filler_items,9)])
        self.semfeatmatrix = np.vstack([pad_with_zero, filler_feats])
        self.lexicalmatrix = np.eye(1579)
        self.orthmatrix = wordlist_to_orth(self.words)


        
        
    
    
        