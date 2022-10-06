#%%

import numpy as np
from scipy.signal import butter, lfilter
  
def pull_vars(data, small_data, vars):
    """Pulls specific variables out of data and into small_data.

    Parameters
    ----------
    data : dict
        The pulled data from the matlab engine. Its large so we only pull specific variables.

    small_data : dict
        Individual subject data that gets appended for every subject.

    vars : list
        List of the variables to pull from data.

    Returns
    ----------
    small_data
        Updated data set.

    """

    for trial in range(len(data)):
        #print(trial)
        trial_key = 'trial_'+str(trial)
        small_data.update({trial_key: {}})
        for obj in vars:
            #print(data[trial].keys())
            small_data[trial_key].update({obj: data[trial][obj]})
        small_data[trial_key].update({'Right_HandVel': \
            np.sqrt(data[trial]['Right_HandXVel']**2 + data[trial]['Right_HandYVel']**2)})
        small_data[trial_key].update({'Right_HandAcc': \
            np.sqrt(data[trial]['Right_HandXAcc']**2 + data[trial]['Right_HandYAcc']**2)})
        small_data[trial_key].update({'Reward_prob': data[trial]['TP_TABLE']['Reward_Probability']})
    return small_data

def squeezin2(data, mltype, n_call):
    """Rercursively squeezes the entire object.

    This is used to remove the matlab data type from any and all parts of the dataset. Recrusively goes into each element and squeezes (numpy) to convert it to a numpy array.

    Parameters
    ----------
    data : list/dict
        The data set to squeeze.

    mltype : data type

    n_call : int

    Returns
    ----------
    data
        Either the sub list/dict or the squeezed data.

    n_call
        Tracker for how many times this gets called.

    """
    if isinstance(data,dict): 
        # Could make better by fixing the next line along with the line after the elif.
        for k, item in enumerate(data):
            if isinstance(data[item],mltype):
                data[item] = np.squeeze(data[item])
            else:
                n_call += 1
                data[item], n_call = squeezin2(data[item], mltype, n_call)
    elif isinstance(data,list):
        for k, item in enumerate(data):
            if isinstance(data[k],mltype):
                data[k] = np.squeeze(data[k])
            else:
                n_call += 1
                data[k], n_call = squeezin2(data[k], mltype, n_call)
    return data, n_call

def filterin(out, filt_vars):
    """Filters using the butterworth filter.

    Parameters
    ----------
    out : list
        The list of things to be squeezed.

    filt_vars : list or dict
        The keys to out.

    Returns
    ----------
    out : list
        The squeezed list.
    """
    for var in filt_vars:
        out[var] = butter_lowpass_filter(out[var], 5, 1000, 5)
    return out

def butter_lowpass_filter(data, cutoff = 5, fs = 1000, order=5):
    """A function to run a butterworth lowpas filter.

    Parameters
    ----------
    data : list
        The data to be filtered.

    cutoff : double
        The cuttoff frequency for the filter (default is 5)

    fs : double
        Sample frequency (default is 1000 for Kinarm)

    Returns
    ----------
    y : list
        The filtered data.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff,
                  btype='low',
                  analog=False)

    y = lfilter(b, a, data)
    return y

def get_v_sign(data):
    """Re assign outward velocity and towards target velocities.

    Parameters
    ----------
    data : list/dict
        Movement data set.

    Returns
    ----------
    data
        Appended dataset with redone velocities.
    """
    v_sign = [v if k < data['vigor']['idx']['move_back'] else v*-1 for k, v in enumerate(data['Right_HandVel'])]
    a_sign = [a if k < data['vigor']['idx']['move_back'] else a*-1 for k, a in enumerate(data['Right_HandAcc'])]
    v_abs = [np.abs(v) for k, v in enumerate(data['Right_HandVel'])]
    a_abs = [a for k, a in enumerate(data['Right_HandAcc'])]
    data.update({'v_sign': np.array(v_sign),
                 'a_sign': np.array(a_sign),
                 'v_abs': np.array(v_abs),
                 'a_abs': np.array(a_abs)})
    return data

def shift_x(data, target_data):
    """Shifts the x and y values to set [0,0] at the home circle.

    It will compute new x and y values of the cursor position so that the home cicle is at [0,0]. All positional data should be off of this now.

    Parameters
    ----------
    data : list
        Positional data.

    target_data : list
        Target data.

    Returns
    ----------
    data : list
        Shifted/transformed positional data.
    """
    x = data['Right_HandX']
    y = data['Right_HandY']
    x_home = target_data['TARGET_TABLE']['X_GLOBAL'][0]/100
    y_home = target_data['TARGET_TABLE']['Y_GLOBAL'][0]/100
    x_shift = x-x_home
    y_shift = y-y_home
    data['X'] = x_shift
    data['Y'] = y_shift
    data['P'] = np.sqrt(x_shift**2+y_shift**2)
    return data

def t_dist(data, target_data):
    """Computes the distance between the hand and target.

    Parameters
    ----------
    data : list
        Positional data.

    target_data : list
        Target data.

    Returns
    ----------
    data : list
        Shifted/transformed positional data.
    """

    if 'X' not in data.keys():
        data = shift_x(data, target_data)

    t_num = int(data['TRIAL']['TP'])
    if t_num>4:
        t_num += -4
    tx = target_data['TARGET_TABLE']['X'][t_num+2]/100
    ty = target_data['TARGET_TABLE']['Y'][t_num+2]/100

    x = data['X']
    y = data['Y']

    data['t_dist'] = np.sqrt((x-tx)**2+(y-ty)**2)
    data['t_diff'] = np.diff(data['t_dist'])/.001
    data['tx'] = tx
    data['ty'] = ty
    data['TRIAL']['TARGET'] = t_num
    return data

def get_mvttimes(data, target_data):
    """Determines the movement time and indexes of when specific events occur within the movement.

    Parameters
    ----------
    data : list
        Positional data.

    target_data : list
        Target data.

    Events
    ----------
    idx_targetshow
        When the target initially appears.

    idx_onset
        When the subject begins moving.

    idx_peakv
        Peak velocity index.

    idx_attarget
        When subject first hits the target.

    idx_moveback
        When the subject begins moving back.

    idx_offset
        When the subject stops moving.

    Returns
    ----------
    vigor : Dict
        Indexes for events.
    """

    data.update({'rad_v': np.diff(data['P']*1000),
                 'px_sign': np.array(data['Right_HandX']),
                 'py_sign': np.array(data['Right_HandY']),
                 'vx_sign': np.array(data['Right_HandXVel']),
                 'vy_sign': np.array(data['Right_HandYVel']),
                 'ax_sign': np.array(data['Right_HandXAcc']),
                 'ay_sign': np.array(data['Right_HandYAcc']),
                 'px_abs': np.abs(data['Right_HandX']),
                 'py_abs': np.abs(data['Right_HandY']),
                 'vx_abs': np.abs(data['Right_HandXVel']),
                 'vy_abs': np.abs(data['Right_HandYVel']),
                 'ax_abs': np.array(data['Right_HandXAcc']),
                 'ay_abs': np.array(data['Right_HandYAcc'])})
    # Shift the data and calculate target locations if you haven't
    if 'X' not in data.keys():
        print('shifting')
        data = shift_x(data, target_data)
    if 't_dist' not in data.keys():
        print('t)disting')
        data = t_dist(data, target_data)

    vigor = {}

    # Pull out variables for ease of use later
    x, y, vx, vy, v = data['X'], data['Y'], data['Right_HandXVel'], data['Right_HandYVel'], data['Right_HandVel'] 

    tx = data['ty']
    ty = data['tx']
    t_diff = data['t_diff']
    P = data['P']

    # Determine index's for spcific points
    vigor.update({'idx': {}})

    # Find target show.
    data['EVENTS']['TIMES'] = np.squeeze(data['EVENTS']['TIMES'])
    for k, item in enumerate(data['EVENTS']['LABELS']):
        if item[0:9] == 'TARGET_ON':
            break
        elif k == len(data['EVENTS']['LABELS']):
            k = -1
    if k != -1:
        idx_targetshow = int(1000*float('%0.2f' % data['EVENTS']['TIMES'][k]))
    else:
        idx_targetshow = 0
    
    # Find movement onset.
    if max(P[idx_targetshow:]) > 0.05:
        b = next(i for i, p in enumerate(P[idx_targetshow:]) if p > .05)
    elif max(P[idx_targetshow:]) > 0.025:
        b = next(i for i, p in enumerate(P[idx_targetshow:]) if p > .025)
    else:
        print('going 3')
        maxp = max(P[idx_targetshow:])
        b = next(i for i, p in enumerate(P[idx_targetshow:]) if p > 0.75*maxp)

    b += idx_targetshow
    a = b
    
    for idx_onset in np.arange(b,50,-1):
        if np.std(t_diff[idx_onset-50:idx_onset])<2e-3 and v[idx_onset]<.03:
            break
    try:
        idx_onset += 0
    except:
        print('Didn\'t find onset')
        idx_onset = b

    # This is a plot checker. Generally uneeded.
    # if np.mean(P[idx_onset-50:idx_onset])>.01:
    #     import matplotlib.pyplot as plt
    #     plt.plot(P)
    #     plt.plot(idx_onset,P[idx_onset],'x',color='red')
    #     plt.show()

    #     plt.plot(x,y,'o')
    #     plt.plot(0,0,'x',markersize=10, color = 'green')
    #     plt.plot(x[idx_onset],y[idx_onset],'x',markersize=10,color = 'red')
    #     plt.show()

    
    # Find at target.
    if max(P) > 0.1:
        idx_attarget = next(i for i, p in enumerate(P[a:-1]) if p > 0.10)+a
    elif min(data['t_dist']) < 0.04:
        idx_attarget = next(i for i, d in enumerate(data['t_dist']) if d < 0.04)
    elif np.argmin(data['t_dist']) - idx_onset>0:
        idx_attarget = np.argmin(data['t_dist'])
    else:
        idx_attarget = np.argmax(P)

    # Find moveback.
    try:
        idx_moveback = next(i for i, d in enumerate(P[idx_attarget:]) if P[i+idx_attarget]<P[i-1+idx_attarget])+idx_attarget
    except:
        idx_moveback = np.argmax(P)

    if idx_moveback < idx_onset:
        idx_moveback = idx_onset + 100

    for idx_offset in range(idx_moveback,len(P)):
        if np.std(t_diff[idx_offset:idx_offset+40])<2e-3 and v[idx_offset]<.03:
            break

    # Find peak velocity.
    idx_peakv = np.argmax(data['rad_v'][idx_onset:idx_moveback])+idx_onset
    idx_retpeakv = np.argmin(data['rad_v'][idx_peakv:len(data['rad_v'])])+idx_peakv

    # Get movement duration
    move_dur = 0.001*(idx_attarget - idx_onset)
    
    # Get reaction time
    react_time = 0.001*(idx_onset-idx_targetshow)
    if react_time < -0.1:
        print('REACT < -0.1 U BORKED IT.')

    # Get peak velocity
    peak_vel = np.max(data['Right_HandVel'][idx_onset:idx_moveback])

    # Return peak velocity
    if idx_offset - idx_moveback > 10:
        peak_vel_moveback = np.max(data['Right_HandVel'][idx_moveback:idx_offset])
    else:
        peak_vel_moveback = np.max(data['Right_HandVel'][idx_moveback-10:idx_offset])

    # Maximum Excursion
    maxex = np.max(P)

    vigor['idx'].update({'onset': idx_onset,
                         'peakv': idx_peakv,
                         'retpeakv': idx_retpeakv,
                         'at_target': idx_attarget,
                         'offset': idx_offset,
                         'target_show': idx_targetshow,
                         'move_back': idx_moveback})

    # Another plot checker.
    # if react_time<.1:
    #     import matplotlib.pyplot as plt
    #     fig, (ax1,ax2) = plt.subplots(2,1)
    #     ax1.plot(data['Right_HandVel'])
    #     ax1.plot(vigor['idx']['onset'],
    #             data['Right_HandVel'][vigor['idx']['onset']],
    #             marker = 'x', color = 'red')
    #     ax2.plot(t_diff)
    #     ax2.plot(vigor['idx']['onset'],
    #             t_diff[vigor['idx']['onset']],
    #             marker = 'x', color = 'red')
    #     plt.show()
    #     blank = 1
        # input("Press Enter to continue...")

    vigor.update({'move_dur': move_dur,
                  'peak_vel': peak_vel,
                  'peak_vel_moveback': peak_vel_moveback,
                  'react_time': react_time,
                  'maxex': maxex})
    return vigor

def est_p(data):
    """Determines estimated probability of reward for each trial/block.

    Calculated the average block probabilities of reward and also uses a perfect observer model to estimate reward probabilities.

    Parameters
    ----------
    data : list/dict
        Subject dataframe to get probabilities of reward for each trial.

    Returns
    ----------
    data
        Updated data set.

    """
    n_shown = [0,0,0,0]
    n_rewarded = [0,0,0,0]
    r_trials = []
    block = 0

    b_trials = [[],[],[],[]]
    block = 0
    for k in range(len(data)):
        if k > 15:
            trial_in_block = (k-16) % 180
            if trial_in_block == 0:
                block += 1
            b_trials[block-1].append(k)

    # Compute the block probailities and add to the dataset.
    # This will only be one of 4 probabilites.
    block = 0
    known_probs = [1, .66, .33, 0]
    block_probs = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    prob_err = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for k, trial in enumerate(data):
        if k > 15:
            trial_in_block = (k-16) % 180
            if trial_in_block == 0:
                block += 1
                n_shown = [0,0,0,0]
                n_rewarded = [0,0,0,0]
            target = int(data[trial]['TRIAL']['TP'])
            n_shown[int(target)-1] += 1
            if len(data[trial]['EVENTS']['LABELS']) >= 3:
                if data[trial]['EVENTS']['LABELS'][3][-4:] == 'getR':
                    n_rewarded[int(target)-1] += 1
                else:
                    r_trials.append(0)
            if trial_in_block == 179:
                for item in known_probs:
                    block_probs[block-1][np.argmin(np.abs(np.array(n_rewarded)/np.array(n_shown)-item))] = item
                    prob_err[block-1][np.argmin(np.abs(np.array(n_rewarded)/np.array(n_shown)-item))] = np.min(np.abs(np.array(n_rewarded)/np.array(n_shown)-item))
    data[trial].update({'block_probs': block_probs,
                        'prob_err': prob_err})

    # Compute the prefect observer probabilites
    block = 0
    for k, trial in enumerate(data):
        if k < 16:
            data[trial].update({'est_prob': 0,
                                'rewarded': 0,
                                't_since_reward': k,
                                'diff_prob': 0,
                                'r_prob': 0,
                                'RPE': 0,
                                'prior_RPE': 0,
                                'prior_RWD': 0})
        else:
            trial_in_block = (k-16) % 180
            if trial_in_block == 0:
                block += 1
            # Estimate the reward using some custom algorithm
            target = int(data[trial]['TRIAL']['TP'])
            n_shown[int(target)-1] += 1
            est_prob = n_rewarded[target-1]/n_shown[target-1]
            if len(data[trial]['EVENTS']['LABELS']) >= 3:
                if data[trial]['EVENTS']['LABELS'][3][-4:] == 'getR':
                    n_rewarded[int(target)-1] += 1
                    r_trials.append(1)
                else:
                    r_trials.append(0)
            else:
                r_trials.append(-1)
            for l in reversed(range(len(r_trials))):
                t_since_reward = len(r_trials)-l
                if r_trials[l] == 1:
                    break
            data[trial].update({'est_prob': est_prob,
                                'rewarded': r_trials[-1],
                                't_since_reward': t_since_reward})

            # Use just straight reward probabilities
            r_prob = block_probs[block-1][data[trial]['TRIAL']['TARGET']-1]

            data[trial].update({'r_prob': r_prob,
                                'diff_prob': r_prob-data['trial_'+str(k)]['r_prob']})

            # RPE
            if len(data[trial]['EVENTS']['LABELS']) >= 3:
                if data[trial]['EVENTS']['LABELS'][3][-4:] == 'getR':
                    RPE = 1-r_prob
                else:
                    RPE = -r_prob
            else:
                RPE = -r_prob

            if RPE == 1 or (RPE == -1 and len(data[trial]['EVENTS']['LABELS']) >= 3):
                print(trial + ' has an r_prob = ' + str(np.round(RPE,2)))

            data[trial].update({'RPE': np.round(RPE,2)})
            data[trial].update({'prior_RPE': data['trial_'+str(k)]['RPE']})
            data[trial].update({'prior_RWD': data['trial_'+str(k)]['rewarded']})
    return data

def calc_errors(data, target_data):
    """Calculate movement error metrics.

    Parameters
    ----------
    data : list/dict
        Movement data set.

    target_data : dict
        Target positions.

    Returns
    ----------
    data
        Appended dataset with error metrics.
    """
    
    # Data should be a single trial
    # Calc endpoint error
    x_cross, y_cross = data['X'][data['vigor']['idx']['at_target']],\
        data['Y'][data['vigor']['idx']['at_target']]
    
    t_num = int(data['TRIAL']['TP'])
    if t_num>4:
        t_num += -4
    tx = target_data['TARGET_TABLE']['X'][t_num+2]/100
    ty = target_data['TARGET_TABLE']['Y'][t_num+2]/100

    # Euclidean Distance Error
    x_err = np.sqrt((x_cross - tx)**2)
    y_err = np.sqrt((y_cross - ty)**2)
    err = x_err = np.sqrt((x_cross - tx)**2+(y_cross-ty)**2)

    # Move back Error
    x_moveback, y_moveback = data['X'][data['vigor']['idx']['move_back']],\
        data['Y'][data['vigor']['idx']['move_back']]

    x_err = np.sqrt((x_moveback - tx)**2)
    y_err = np.sqrt((y_moveback - ty)**2)
    move_back_err = x_err = np.sqrt((x_moveback - tx)**2+(y_moveback-ty)**2)

    data['vigor'].update({'x_err': x_err,
                          'y_err': y_err,
                          'error_dist': err,
                          'move_back_error': move_back_err})
        
    # Angular Error
    reach_angle = np.arctan2(x_cross, y_cross) * 180 / np.pi
    target_angle = np.arctan2(tx, ty) * 180 / np.pi
    if reach_angle<0:
        reach_angle += 360
    if target_angle<0:
        target_angle += 360

    error_angle = target_angle-reach_angle

    data['vigor'].update({'reach_angle': reach_angle,
                          'target_angle': target_angle,
                          'error_angle': error_angle})

    return data

def traj_check(data):
    """Checker function to make sure trajectories make sense.

    Plots movement trajectories along with indexes of events. Mainly used to check that the movement is valid.

    Parameters
    ----------
    data : list/dict
        Movement data set.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, [ax1, ax2, ax3] = plt.subplots(3, figsize = (9,9))
    p_index = [data['vigor']['idx']['target_show'],
                data['vigor']['idx']['offset']]

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Onset',
                       markerfacecolor='r', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='At Target',
                       markerfacecolor='g', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Offset',
                       markerfacecolor='b', markersize=10)]
    
    vars = ['P','Right_HandXVel','Right_HandXAcc']
    ylabs = ['Position','Velocity ($m/s$)','Acceleration ($m/s^2$)']
    for k, ax in enumerate([ax1,ax2,ax3]):
        ax.plot(data[vars[k]][p_index[0]:p_index[1]+200])
        ax.plot(data['vigor']['idx']['onset']-p_index[0],
                data[vars[k]][data['vigor']['idx']['onset']],
                'o',
                color='red',
                markersize = 10)
        ax.plot(data['vigor']['idx']['at_target']-p_index[0],
                data[vars[k]][data['vigor']['idx']['at_target']],
                'o',
                color='blue',
                markersize = 10)
        ax.plot(data['vigor']['idx']['offset']-p_index[0],
                data[vars[k]][data['vigor']['idx']['offset']],
                'o',
                color='green',
                markersize = 10)
        if k == 0:
            ax.legend(handles = legend_elements, loc = 'lower right')
        ax.set(xlabel='Time (s)', ylabel = ylabs[k])
        ax.set_ylim([min(data[vars[k]])*1.05-.02,max(data[vars[k]])*1.05])
        ax.tick_params(direction = 'out', top = 0, right = 0)
        ax.set_xticklabels(ax1.get_xticks()/1000)

    plt.tight_layout()

def squeezin(out, objs):
    """Squeezes a list or dictionary and all elements underneath. Not used anymore, recursive squeezin2 is uesd.

    Parameters
    ----------
    out : list
        The list of things to be squeezed.

    objs : list or dict
        The keys to out.

    Returns
    ----------
    out : list
        The squeezed list.
    """
    for obj in objs:
        out[obj] = np.squeeze(out[obj])
    return out

# %%
