#!"C:\Users\rjc5t\AppData\Local\Programs\Python\Python38\python.exe"

#The only manual edit necessary is to have the preceeding line
#be the full path to your python 3.6, 3.7, or 3.8 executable.

"""
cf_pull_data.py
--------------------------------------------

This script is used to pull the data from the .zip files saved in the
experiment folder. It uses the matlab python engine to pull the data from the
zip files and then uses python to squeeze the data into a numpy compatible
format.

"""
#1.5 change:
#no longer appending previously saved .pickle data into the cfdata and target_data arrays
#only loads files as needed
#can save missing .mat files that were pickled for whatever reason
#overall coding efficiency improvements
# - C.K.

#%%
import matlab.engine as ml
import os
from os import makedirs, path
import glob
import sys
import numpy as np
from scipy.signal import butter, lfilter
import math
# import seaborn as sns
import pickle
import time
import scipy.io as spio

from cf_functions import *

def pull_mat(filename):
    #takes the subject-specific file path and pulls from either
    #existing .mat file from previous extraction, or runs the KINARM script
    #to pull experiment data from the .kinarm file
    if not(path.exists(filename+".mat")):
        _ = eng.exam_save(os.path.join("Data",filename))

    #need to load in from the experiment rather than "Data" director...
    out = spio.loadmat(filename+".mat",simplify_cells=True)
    print(filename,".mat loaded")
    out = out['c3dstruct']['c3d']
    cfdata= pull_vars(out, {}, obj_need)
    cfdata['trial_1']['filename'] = filename 
    
    target_data = {'BLOCK_TABLE': out[0]['BLOCK_TABLE'],
                        'TP_TABLE': out[0]['TP_TABLE'],
                        'TARGET_TABLE': out[0]['TARGET_TABLE']}
    return(cfdata,target_data)




abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append('Kinarm Analysis Scripts - 3.1.1')

#Directory structure should be as follows:
# current directory/
# ├─ {exp_name}/
# │  ├─ Data/
# │  │  ├─ {dexterit-e data}.kinarm

exp_name = ''
while len(exp_name)==0:
    exp_name = input("Experiment directory to be loaded (case sensitive): ")

save_data = 1
cfdata = []
target_data = []

file_name = 'vigor_conf_postsqueeze_'+exp_name+'.pickle'
KINARM_DIR = os.path.join(dname,'Kinarm Analysis Scripts - 3.1.1')
EXP_DIR = os.path.join(dname,exp_name)
DATA_DIR = os.path.join(EXP_DIR,"Data")

# Switch to data folder and see if there's already pulled data. If pulled data
# already load it.
if not(os.path.isdir(DATA_DIR)):
    os.mkdir(DATA_DIR)

#going into the data folder
os.chdir(DATA_DIR)
already_pulled = []
pulled_pickles = []
missing_mats = []

mat_files = glob.glob("*.mat")
kin_files = glob.glob("*.kinarm")
pick_files = glob.glob("*.pickle")
zip_files = []

#check if pickle data already exists
if any(pick_files):
    #separate pickle files are saved per subject
    print("Loading existing .pickle data")
    for fp in pick_files:
        #unfortunately, we need to load the whole pickle file to determine what source .kinarm file
        #was used to make it.
        with open(fp,'rb') as f:
            temp = pickle.load(f)
        fn = os.path.basename(temp[0]['trial_1']['filename'])       
        del temp
        already_pulled.append(fn)
        pulled_pickles.append(fp)
        print("\tPickle File: "+fp)
        print("\tKinarm File: ",fn)
else:
    zip_files = kin_files

for k_file in kin_files:
        if (k_file not in already_pulled) & (k_file not in zip_files):
            zip_files.append(k_file)

#check if .mat data already exists:
if any(mat_files):
    #compare between these and kin files
    #only include missing .mat files from .kin files
    for k_file in kin_files:
        mat_base = map(lambda fn: fn[0:-4],mat_files)
        if k_file not in mat_base:
            missing_mats.append(k_file)
        
else:
    missing_mats = kin_files
    
#coming back out to experiment folder
os.chdir('..')


#%%
eng = ml.start_matlab()

#add KINARM code path to the matlab engine
eng.addpath(KINARM_DIR)

# Definte objects to pull from zip files.
obj_need = ['Right_HandX', 'Right_HandY', 'Right_HandXVel', 'Right_HandYVel',
            'Right_HandXAcc', 'Right_HandYAcc',
            'Right_FS_TimeStamp', 'HAND', 'EVENTS',
            'EVENT_DEFINITIONS', 'TRIAL','TP_TABLE']



#Loading and/or building the .mat files containing subject's experimental data
for filename in kin_files:
    rel_fp = os.path.join(DATA_DIR,filename)

    #if already existing, MATLAB function only loads the .mat file
    #otherwise, it's built.
    if (filename in missing_mats) | (filename in zip_files):
        cf,td = pull_mat(rel_fp)
    
    #only append to the "to-be-pickled" data array if it needs to be pickled
    if filename in zip_files:
        cfdata.append(cf)
        target_data.append(td)

#back into data folder for experiment
os.chdir(DATA_DIR)

#kill the matlab engine. not needed any more
eng.quit()

#%%

#we only need to pickle data if there is data to be pickled
if cfdata:
    mltype = type(cfdata[0]['trial_1']['Right_HandVel'])

    # Definte variables that need to be squeezed. to convert to the numpy format.
    squeeze_vars = ['Right_HandX', 'Right_HandY','Right_HandVel',
                    'Right_HandXVel', 'Right_HandYVel','Right_HandXAcc', 'Right_HandYAcc', 'Right_FS_TimeStamp']

    # Squeeze the data.
    for i in range(len(cfdata)):
        print("Squeezing File: "+str(i))
        cfdata[i],n_call = squeezin2(cfdata[i],mltype,0)
        print('Number of Calls: ' + str(n_call))
        target_data[i],_ = squeezin2(target_data[i],mltype,0)

    #%%
    # Save the data.
    if save_data:
        max_int = 0
        for s in range(len(cfdata)):
            #get first unused integer from pickle files so that file numbering won't conflict
            #technically the length of "pulled_pickles" should be sufficient, but dumb shit happens
            for fname in pulled_pickles:
                len_name = len(exp_name)
                i1 = fname.find(exp_name)+len_name+1
                i2 = fname.find('.pickle')
                num = int(fname[i1:i2])
                max_int = max(num,max_int)
            f_num = s+max_int+1

            #saving the data for real this time
            file_name = 'vigor_conf_postsqueeze_'+exp_name+'_'+str(f_num)+'.pickle'
            with open(file_name, 'wb') as f:
                pickle.dump([cfdata[s], target_data[s], file_name], f)
                print('Post Squeeze data saved as: '+file_name)