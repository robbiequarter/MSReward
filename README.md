# KinarmCode
Repository for files needed or helpful for data extraction from the kinarm bot

Included are as of now files to extract useful .mat files and .pickle files containing numpy arrays.
To run cf_pull_data, you'll need the system requirements documented [here](https://www.mathworks.com/help/matlab/matlab_external/system-requirements-for-matlab-engine-for-python.html).

Directory structure should be as follows:

  current directory/  
  ├─ cf_pull_data.py  
  ├─ cf_functions.py  
  ├─ {exp_name}/  
  │  ├─ Data/  
  │  │  ├─ {dexterit-e data}.kinarm  
  ├─ Kinarm Analysis Scripts - 3.1.1  
  │  ├─ {lots of other files}  
  │  ├─ exam save.mat  
