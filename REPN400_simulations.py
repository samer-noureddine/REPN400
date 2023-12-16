import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
from scipy.stats import ttest_rel
from PredictiveCoding_Model import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

random.seed(1)
np.random.seed(1)


def run_simulation(**kwargs):
    # run a simulation and keep only the components needed for plotting and data analysis (to avoid out-of-memory errors)
    full_simulation = Simulation(**kwargs)
    data,fname,sim_input = full_simulation.simulation_data, full_simulation.sim_filename, full_simulation.sim_input
    del full_simulation
    return {"simulation_data" : data,
             "sim_filename" : fname,
             "sim_input": sim_input}

def spatial_rsa_within(data):
    '''data: 3D matrix: trial*voxel*time
    return: within-condition RSA'''
    spatial_rsa_withinR = []
    for iter in range(data.shape[-1]):
        corr_matrix = np.corrcoef(data[:,:,iter])
        tril_inds = np.tril_indices(corr_matrix.shape[0],-1)
        nonzero_matrices = corr_matrix[tril_inds]
        spatial_rsa_withinR.append(np.mean(nonzero_matrices))
    withinR = np.array(spatial_rsa_withinR)
    return withinR

def spatial_rsa_btw(data1, data2):
    '''data: 3D matrix: trial*voxel*time
    return: between-condition RSA'''
    spatial_rsa_betweenR = []
    for iter in range(data1.shape[-1]):
        corr_matrix_between = np.corrcoef(data1[:,:,iter],data2[:,:,iter])
        spatial_rsa_betweenR.append(np.mean(corr_matrix_between[:120,120:]))
    btwR = np.array(spatial_rsa_betweenR)
    return btwR


def tempRSA_withintrl(data):
    '''data: 3D matrix: chan*trial*time'''
    tempR = []
    for itrl in range(data.shape[1]):
        pre = data[:,itrl,8:17]
        post = data[:,itrl,21:30]
        tempR.append(np.corrcoef(pre.flatten(),post.flatten())[0,1])
    R = np.array(tempR)
    return R

def FWHM_simple(y):
    """
    Calculate the Full Width at Half Maximum (FWHM) for a given curve.

    Parameters:
    x (numpy array): The x values of the curve.
    y (numpy array): The y values of the curve.
    
    Returns:
    float: The FWHM of the curve and the indices where it occurs
    """
    ymax = np.max(y)
    imax = np.argmax(y)
    yhalf = ymax / 2
    left_idx = np.where(y[:imax] < yhalf)[0][-1]
    right_idx = np.where(y[imax:] < yhalf)[0][0] + imax
    x = np.arange(y.shape[0])
    fwhm = x[right_idx] - x[left_idx]
    return fwhm, left_idx, right_idx


# ---------------------------------------------------------------------------
# Run model
# ---------------------------------------------------------------------------
lexicon = Lexicon()
NUM_ITERS = 20

with open(r'./helper_txt_files/1579words_words.txt') as f:
    w = f.read()
    wordlist = w.split('\n')

with open(r'./helper_txt_files/unrepeated_matlab.csv') as csvfile:
    unrep = pd.read_csv(csvfile)
    colname = unrep.columns[0]
    unrepeated_indices = np.array(unrep[colname]) -1

# ---------------------------------------------------------------------------
##################### DEFINE STIMULI FOR EACH CONDITION ##################### 
# ---------------------------------------------------------------------------
num_trials = 120

standard_stims = wordlist[:512]
# shuffle the standard_stims using the randomly generated unrepeated_indices (using a particular set of indices for reproducibility) 
unrepeated_stims = list(np.array(standard_stims)[unrepeated_indices])

# keep only the first 120 inputs
standard_stims = standard_stims[:num_trials]
unrepeated_stims = unrepeated_stims[:num_trials]

# Pre-activate expected words
cloze_simulations_preactivate = run_simulation(sim_input = standard_stims, 
                                                clamp_iterations =NUM_ITERS,
                                                BU_TD_mode = "top_down",
                                                cloze = 0.99,
                                                sim_filename = f'cloze_simulations_preact_high_cloze')

# this is the EXPECTED simulation
cloze_simulations_bottomup = run_simulation(sim_input = standard_stims, 
                                            clamp_iterations =NUM_ITERS,
                                            BU_TD_mode = "bottom_up", 
                                            prevSim = cloze_simulations_preactivate,
                                            sim_filename = f'cloze_simulations_bottomup_highcloze')

# this is the UNEXPECTED simulation
lexviol_bottomup = run_simulation(sim_input = unrepeated_stims, 
                                            clamp_iterations =NUM_ITERS,
                                            BU_TD_mode = "bottom_up", 
                                            prevSim = cloze_simulations_preactivate,
                                            sim_filename = f'lexviol_bottomup_highcloze')


# ---------------------------------------------------------------------------
# Get data of 120 trials from PC model output
# ---------------------------------------------------------------------------

# get source-level data

expected_word_lex_st = cloze_simulations_bottomup['simulation_data']['all_lex_states'][:,:num_trials,:]
expected_word_lex_pe = cloze_simulations_bottomup['simulation_data']['all_lex_PE'][:,:num_trials,:]
expected_word_sem_st = cloze_simulations_bottomup['simulation_data']['all_sem_states'][:,:num_trials,:]
expected_word_sem_pe = cloze_simulations_bottomup['simulation_data']['all_sem_PE'][:,:num_trials,:]

unexpected_word_lex_st = lexviol_bottomup['simulation_data']['all_lex_states'][:,:num_trials,:]
unexpected_word_lex_pe = lexviol_bottomup['simulation_data']['all_lex_PE'][:,:num_trials,:]
unexpected_word_sem_st = lexviol_bottomup['simulation_data']['all_sem_states'][:,:num_trials,:]
unexpected_word_sem_pe = lexviol_bottomup['simulation_data']['all_sem_PE'][:,:num_trials,:]

# ----------------------------------------------------------------------------------------
# Univariate effects: expected < expected
# FIGURE 1B and FIGURE S1
# ----------------------------------------------------------------------------------------
# get summed activity of PE

exp_lexsem_pe_sum = (np.sum(expected_word_lex_pe,0) + np.sum(expected_word_sem_pe,0))/2 #trial*time
unexp_lexsem_pe_sum = (np.sum(unexpected_word_lex_pe,0) + np.sum(unexpected_word_sem_pe,0))/2
exp_lexsem_st_sum = (np.sum(expected_word_lex_st,0) + np.sum(expected_word_sem_st,0))/2 #trial*time
unexp_lexsem_st_sum = (np.sum(unexpected_word_lex_st,0) + np.sum(unexpected_word_sem_st,0))/2

mean_expected_value = np.mean(exp_lexsem_pe_sum[:,21:30])
mean_unexpected_value = np.mean(unexp_lexsem_pe_sum[:,21:30])
std_expected = np.std(exp_lexsem_pe_sum[:,21:30])
std_unexpected = np.std(unexp_lexsem_pe_sum[:,21:30])
print(f'#expected: {mean_expected_value: .3f} +/- {std_expected: .3f}',
      f'\n#unexpected: {mean_unexpected_value: .3f} +/- {std_unexpected: .3f}')
#expected:  15.188 +/-  15.355 
#unexpected:  146.959 +/-  58.351

t_statistic, p_value = ttest_rel(np.mean(unexp_lexsem_pe_sum[:,21:30],1), np.mean(exp_lexsem_pe_sum[:,21:30],1))
print("#T-statistic:", t_statistic)
print("#P-value:", p_value)
#T-statistic: 86.1017573136251
#P-value: 4.828104003683159e-109

mean_expected_timecourse_pe = np.mean(exp_lexsem_pe_sum, axis = 0)
mean_unexpected_timecourse_pe = np.mean(unexp_lexsem_pe_sum, axis = 0)
mean_expected_timecourse_st = np.mean(exp_lexsem_st_sum, axis = 0)
mean_unexpected_timecourse_st = np.mean(unexp_lexsem_st_sum, axis = 0)


#plot raw PE activity
plt.figure(figsize = (12,4))
plt.subplot(121)
plt.plot(mean_unexpected_timecourse_pe,'r--',label = 'Error: Unexpected')
plt.plot(mean_expected_timecourse_pe ,'r-',label = 'Error: Expected')
plt.plot(mean_expected_timecourse_st ,'b-',label = 'State: Expected')
plt.plot(mean_unexpected_timecourse_st,'b--',label = 'State: Unexpected')
plt.xlim(15,41)
plt.ylim(-5,220)
plt.legend()
plt.xlabel('Number of Iterations')
plt.ylabel('Raw Lexico-semantic Prediction Error')

plt.subplot(122)
plt.plot(mean_unexpected_timecourse_pe/np.max(mean_unexpected_timecourse_pe),'r--',label = 'Error: Unexpected')
plt.plot(mean_expected_timecourse_pe/np.max(mean_unexpected_timecourse_pe) ,'r-',label = 'Error: Expected')
plt.plot(mean_expected_timecourse_st/np.max(mean_expected_timecourse_st) ,'b-',label = 'State: Expected')
plt.plot(mean_unexpected_timecourse_st/np.max(mean_expected_timecourse_st),'b--',label = 'State: Unexpected')
plt.xlim(15,41)
plt.ylim(-0.05,1.1)
plt.legend()
plt.xlabel('Number of Iterations')
plt.ylabel('Normalized Lexico-semantic Prediction Error')
if not os.path.exists('./plots/'):
    os.mkdir('./plots/')
plt.savefig('./plots/FigS1_univariate_summed_lexsem_raw.eps', format='eps', dpi=300)
plt.savefig('./plots/FigS1_univariate_summed_lexsem_raw.png')
plt.savefig('./plots/FigS1_univariate_summed_lexsem_raw.png')

#plot difference activity
plt.figure()
plt.plot(mean_unexpected_timecourse_pe - mean_expected_timecourse_pe, label = 'Unexpected minus expected')
plt.xlim(15,41)
plt.ylim(-5,220)
plt.legend()
plt.xlabel('Number of Iterations')
plt.ylabel('Lexico-semantic Prediction Error')
plt.savefig('./plots/Fig1B_univariate_summed_lexsem_diff.eps', format='eps', dpi=300)
plt.savefig('./plots/Fig1B_univariate_summed_lexsem_diff.png')
plt.savefig('./plots/Fig1B_univariate_summed_lexsem_diff.png')

# ----------------------------------------------------------------------------------------
# Spatial RSA: within-expected > between-condition
# FIGURE 3C and FIGURE S2
# ----------------------------------------------------------------------------------------
# get summed activity of PE and state separately
exp_lexsem_st_sum = (np.sum(expected_word_lex_st,0) + np.sum(expected_word_sem_st,0))/2 #trial*time
exp_lexsem_pe_sum = (np.sum(expected_word_lex_pe,0) + np.sum(expected_word_sem_pe,0))/2
unexp_lexsem_st_sum = (np.sum(unexpected_word_lex_st,0) + np.sum(unexpected_word_sem_st,0))/2
unexp_lexsem_pe_sum = (np.sum(unexpected_word_lex_pe,0) + np.sum(unexpected_word_sem_pe,0))/2

exp_lexsem_stpe = np.stack((exp_lexsem_st_sum,exp_lexsem_pe_sum),axis=2)#trial*time*chan
unexp_lexsem_stpe = np.stack((unexp_lexsem_st_sum,unexp_lexsem_pe_sum),axis=2)
exp_stpe = np.transpose(exp_lexsem_stpe, (0, 2, 1)) #trial*chan*time
unexp_stpe = np.transpose(unexp_lexsem_stpe, (0, 2, 1))

np.random.seed(25)
nvoxels = 20
num_trials= 120
chans = exp_stpe.shape[1]
mapping_matrix = np.random.randint(1,10,(nvoxels,chans)) #voxel*chan
noise = np.random.normal(0,3,(nvoxels, num_trials)) #voxel*trial
MEG_expected = np.zeros((num_trials, nvoxels,41))
MEG_unexpected = np.zeros((num_trials, nvoxels,41))
for iter in range(exp_stpe.shape[-1]):
    MEG_expected[:,:,iter] = np.dot(mapping_matrix, exp_stpe[:,:,iter].T).T  + noise.T #trial*voxel*time
    MEG_unexpected[:,:,iter] = np.dot(mapping_matrix, unexp_stpe[:,:,iter].T).T + noise.T

spatial_rsa_within_expected = spatial_rsa_within(MEG_expected)
spatial_rsa_within_unexpected = spatial_rsa_within(MEG_unexpected)
spatial_rsa_between_exp_unexp = spatial_rsa_btw(MEG_expected,MEG_unexpected)

# plot difference waves
plt.figure()
plt.plot(spatial_rsa_within_expected - spatial_rsa_between_exp_unexp, label = 'within-expected minus between-condition')
plt.xlim(15,41)
plt.ylim(-0.01,0.10)
plt.legend()
plt.savefig('./plots/Fig3C_spatRSA_summed_lexsem_dif.eps', format='eps', dpi=300)
plt.savefig('./plots/Fig3C_spatRSA_summed_lexsem_dif.png')


# get summed activity of lexical PE, lexical ST, semantic PE and semantic ST separately
exp_lex_st_sum = np.sum(expected_word_lex_st,0) #trial*time
exp_lex_pe_sum = np.sum(expected_word_lex_pe,0)
unexp_lex_st_sum = np.sum(unexpected_word_lex_st,0)
unexp_lex_pe_sum = np.sum(unexpected_word_lex_pe,0)

exp_sem_st_sum = np.sum(expected_word_sem_st,0) #trial*time
exp_sem_pe_sum = np.sum(expected_word_sem_pe,0)
unexp_sem_st_sum = np.sum(unexpected_word_sem_st,0)
unexp_sem_pe_sum = np.sum(unexpected_word_sem_pe,0)

exp_lex_stpe = np.stack((exp_lex_st_sum,exp_lex_pe_sum),axis=2)#trial*time*chan
unexp_lex_stpe = np.stack((unexp_lex_st_sum,unexp_lex_pe_sum),axis=2)
exp_lex_stpe = np.transpose(exp_lex_stpe, (0, 2, 1)) #trial*chan*time
unexp_lex_stpe = np.transpose(unexp_lex_stpe, (0, 2, 1))

exp_sem_stpe = np.stack((exp_sem_st_sum,exp_sem_pe_sum),axis=2)#trial*time*chan
unexp_sem_stpe = np.stack((unexp_sem_st_sum,unexp_sem_pe_sum),axis=2)
exp_sem_stpe = np.transpose(exp_sem_stpe, (0, 2, 1)) #trial*chan*time
unexp_sem_stpe = np.transpose(unexp_sem_stpe, (0, 2, 1))

exp_stpe = {
    'lex':exp_lex_stpe,
    'sem':exp_sem_stpe
}
unexp_stpe = {
    'lex':unexp_lex_stpe,
    'sem':unexp_sem_stpe
}


plt.figure(figsize = (12,4))
for i,level in enumerate(['lex', 'sem']):
    nvoxels = 20
    num_trials= 120
    chans = exp_stpe[level].shape[1]
    MEG_expected = np.zeros((num_trials, nvoxels,41))
    MEG_unexpected = np.zeros((num_trials, nvoxels,41))
    for iter in range(exp_stpe[level].shape[-1]):
        MEG_expected[:,:,iter] = np.dot(mapping_matrix, exp_stpe[level][:,:,iter].T).T  + noise.T #trial*voxel*time
        MEG_unexpected[:,:,iter] = np.dot(mapping_matrix, unexp_stpe[level][:,:,iter].T).T + noise.T

    spatial_rsa_within_expected = spatial_rsa_within(MEG_expected)
    spatial_rsa_within_unexpected = spatial_rsa_within(MEG_unexpected)
    spatial_rsa_between_exp_unexp = spatial_rsa_btw(MEG_expected,MEG_unexpected)

    # plot difference waves
    plt.subplot(121+i)
    plt.plot(spatial_rsa_within_expected - spatial_rsa_between_exp_unexp)
    plt.xlim(15,41)
    if level == 'lex':
        plt.ylim(-0.01,0.4)
        plt.title('Simulated cross-trial similarity effect\nLexical')
        plt.ylabel('Within-expected minus between')
    else:
        plt.ylim(-0.01,0.1)
        plt.title('Simulated cross-trial similarity effect\nSemantic') 
        
    # Calculate FWHM and indices
    within_minus_between = spatial_rsa_within_expected - spatial_rsa_between_exp_unexp
    timecourse_indices = np.arange(within_minus_between.shape[0])
    fwhm, left_idx, right_idx = FWHM_simple(within_minus_between)
    # Indicating the FWHM region
    plt.axhline(y=within_minus_between[left_idx], color='r', linestyle='--', label='FWHM')
    plt.axvline(x=timecourse_indices[left_idx], color='g', linestyle='--')
    plt.axvline(x=timecourse_indices[right_idx], color='g', linestyle='--')

plt.savefig('./plots/FigS2_spatRSA_summed_lex_and_sem_dif.eps', format='eps', dpi=300)
plt.savefig('./plots/FigS2_spatRSA_summed_lex_and_sem_dif.png')

# -------------------------------------------------------------------------------------------------
# Temporal RSA: expected > unexpected
# FIGURE 2D
# -------------------------------------------------------------------------------------------------
# extract lexico-semantic state activity, run within-trial tempRSA
exp_lexsem = np.concatenate((expected_word_lex_st,expected_word_sem_st),axis=0) #chan*trial*time
unexp_lexsem = np.concatenate((unexpected_word_lex_st,unexpected_word_sem_st),axis=0)
tempR_expected_lexsem = tempRSA_withintrl(exp_lexsem) #input: chan*trial*time
tempR_unexpected_lexsem = tempRSA_withintrl(unexp_lexsem)

#-------------------------------------------------
# Box plot: lexico-semantic
plotdata = [tempR_expected_lexsem, tempR_unexpected_lexsem]
labels = ['Expected', 'Unexpected']
plt.figure()
plt.boxplot(plotdata, labels=labels)
plt.xlabel('Condition')
plt.ylabel('Temporal RSA')
plt.title('temporal similarity at the lexico-semantic level')
plt.savefig('./plots/Fig2D_tempRSA_boxplot_lexsem_fullspace.eps', format='eps', dpi=300)
plt.savefig('./plots/Fig2D_tempRSA_boxplot_lexsem_fullspace.png')
plt.show()

#-------------------------------------------------
# stats
mean_expected = np.mean(tempR_expected_lexsem)
mean_unexpected = np.mean(tempR_unexpected_lexsem)
std_expected = np.std(tempR_expected_lexsem)
std_unexpected = np.std(tempR_unexpected_lexsem)
print(f'expected: {mean_expected: .3f} +/- {std_expected: .3f}',
      f'\nunexpected: {mean_unexpected: .3f} +/- {std_unexpected: .3f}')
#expected:  0.637 +/-  0.019 
#unexpected:  0.088 +/-  0.072

t_statistic, p_value = ttest_rel(tempR_expected_lexsem, tempR_unexpected_lexsem)
print("#T-statistic:", t_statistic)
print("#P-value:", p_value)
#T-statistic: 86.77782421946165
#P-value: 1.931296402350636e-109


#-----------------------------------------------------------------------------
# Plot unique temporal patterns for each item
# Use MDS to extract the time course in the reduced spatial dimensions
#-----------------------------------------------------------------------------
# FIGURE 2C
#------------------------------------------------------------------
# MDS for the pre and post windows separately

n_components = 3  # Number of dimensions to reduce to
mds = MDS(n_components=n_components, random_state=42)
scaler = MinMaxScaler()

def get_3d_seg(data,time_range):
    get_3d_list = []
    normalized_data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    for item in range(data.shape[1]):    
        reduced_data_3d = mds.fit_transform(normalized_data[:, item, time_range].T) #time*chan
        get_3d_list.append(reduced_data_3d)
    get_3d_data = np.array(get_3d_list)#trial*time*chan
    newdata = np.transpose(get_3d_data,(2,0,1))
    return newdata

# concatenated lexico-semantic state activity
time_range = slice(21, 30) 
expected_lexsem_post = get_3d_seg(exp_lexsem,time_range)
unexpected_lexsem_post = get_3d_seg(unexp_lexsem,time_range)
time_range = slice(8, 17) 
expected_lexsem_pre = get_3d_seg(exp_lexsem,time_range)


#------------------------------------------------------------------
# plot temporal patterns in 3D space

def plot_3d_data(ax, data, colors, item_range, time_range, title, xlim=None, ylim=None, zlim=None):
    for ind, item in enumerate(item_range):
        plotdata = data[:, item, time_range].T
        interp_functions = [interp1d(np.arange(len(plotdata)), plotdata[:, i], kind='cubic') for i in range(3)]
        num_interp_points = 200
        interp_indices = np.linspace(0, len(plotdata) - 1, num_interp_points)
        interp_data = np.column_stack([interp_function(interp_indices) for interp_function in interp_functions])

        ax.plot(interp_data[:, 0], interp_data[:, 1], interp_data[:, 2], linestyle='-', color=colors[ind], label=f'Item {item + 1}')

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0.1)
    ax.view_init(elev=20, azim=30)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

fig_pre = plt.figure(figsize = (12,8))
ax_pre = fig_pre.add_subplot(111, projection='3d')
fig_post_exp = plt.figure(figsize = (12,8))
ax_post_exp = fig_post_exp.add_subplot(111, projection='3d')
fig_post_unexp = plt.figure(figsize = (12,8))
ax_post_unexp = fig_post_unexp.add_subplot(111, projection='3d')
colors = ['red', 'blue', 'cyan', 'purple','magenta', 'brown', 'pink', 'gray', 'orange', 'green']
item_range = [60, 110,  10]
plot_3d_data(ax_pre, expected_lexsem_pre, colors,item_range, range(expected_lexsem_pre.shape[2]), 'lexico-semantic level: Pre-activated')
plot_3d_data(ax_post_exp, expected_lexsem_post, colors, item_range, range(expected_lexsem_post.shape[2]), 'lexico-semantic level: Expected')
plot_3d_data(ax_post_unexp, unexpected_lexsem_post, colors, item_range, range(unexpected_lexsem_post.shape[2]), 'lexico-semantic level: Unexpected')

fig_pre.savefig('./plots/Fig2C_separatewindows_pre_lexsem.eps', format='eps', dpi=300, bbox_inches='tight')
fig_pre.savefig('./plots/Fig2C_separatewindows_pre_lexsem.png')
fig_post_exp.savefig('./plots/Fig2C_separatewindows_post_exp_lexsem.eps', format='eps', dpi=300, bbox_inches='tight')
fig_post_exp.savefig('./plots/Fig2C_separatewindows_post_exp_lexsem.png')
fig_post_unexp.savefig('./plots/Fig2C_separatewindows_post_unexp_lexsem.eps', format='eps', dpi=300, bbox_inches='tight')
fig_post_unexp.savefig('./plots/Fig2C_separatewindows_post_unexp_lexsem.png')
plt.show()
