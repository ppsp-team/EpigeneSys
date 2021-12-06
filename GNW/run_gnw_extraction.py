#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : run_gnw_extraction.py
# author          : Konstantin Volzhenin, Guillaume Dumas
# date            : 2021-04-06
# version         : 3
# usage           : python run_gnw_extraction.py
# notes           : need the file "spikes_for_global_alt.pickle"
# python_version  : 3.7
# ==============================================================================


import numpy as np
from copy import copy
import pickle
import os
import pandas as pd


def extract_data(filename):
    with open(filename, 'rb') as f:
        dump = pickle.load(f)

    # dump = [input_mon.t / second, input_mon.i * 1, dopamine_monitor.t / second, dopamine_monitor.i * 1, N_activity_plot,
    #             synapse_stdp_monitor_out.t/second, synapse_stdp_monitor_out.s.T/gmax]

    # Input spikes
    inpt = dump[0]
    inpi = dump[1]

    # Extracellular dopamine
    dopt = dump[2]
    dopi = dump[3]

    # Number of neurons per group on a graph
    N_activity_plot = dump[4]

    # GNW-Motor Cortex synaptic weights
    synt = dump[5]
    synw = dump[6]

    # Total runtime to use in the analysis. If you decide to produce data for different time periods, please change
    # this number to a corresponding total length of your simulation
    runtime = 350

    reward = np.floor(dopt)
    zero_shown = inpt[inpi == 0][::50]
    negative_reward = np.asarray([x for x in reward if x in zero_shown])

    one_shown = inpt[inpi == 1][::50]
    missed_positive_reward = np.asarray([x for x in one_shown if x not in reward])

    bins = np.linspace(0, runtime, num=10)
    matured_value = runtime / len(bins) * 0.1

    # The first runtime - T seconds are train data, the last T seconds are used as test data
    T = 100.

    # calculated for last T seconds out of total 350s run
    score = 100 - (len(negative_reward[negative_reward > (runtime - T)]) +
                   len(missed_positive_reward[missed_positive_reward > (runtime - T)])) * 100 / T

    w_fin = synw[-1]
    n_selected = len(w_fin[w_fin >= 0.5])
    n_pruned = len(w_fin[w_fin < 0.5])

    variance = np.var(synw, axis=1)
    var_fin = np.mean(variance[np.where(synt == (runtime - T))[0][0]:])

    learn_time = np.nan
    
    meanw = np.mean(synw, axis=1)
    
    y = np.poly1d(np.polyfit(synt,meanw,15))(synt)
    diff = np.diff(y)
    epsilon = np.mean(np.abs(diff)) * 0.1
    
    count = 0
    for t in range(len(synt) - 1):
        if np.abs(diff[t]) < epsilon:
            count += 1
        else:
            count = 0
        if count > 100:
            learn_time = synt[t - count]
            break

    return score, var_fin, learn_time, n_selected, n_pruned


# For every run we save following parameters:
# # Number of a simulation
# # Level of the spontaneous intrinsic activity
# # Excitatory/Inhibitory ratio in the GNW
# # Conditioning type (True - Trace, False - Delay)
# # Final score (based on the last T seconds)
# # Final variance of the GNW-MC weights
# # Learning time (if applicable)
# # Number of selected neurons, as well as the number of pruned neurons


print("sim_num,rate_num,ratio,trace,score,var_fin,learn_time,n_selected,n_pruned")
accu = []
for sim_num in range(10):
    for rate_num in range(11):
        for ratio in range(5, 101, 5):
            for trace in [True, False]:  # False = Delay conditioning
                filename = 'results/data{}_rate{}_ratio{}_cond{}.pickle'.format(sim_num, rate_num, ratio, trace)
                if os.path.exists(filename):
                    score, var_fin, learn_time, n_selected, n_pruned = extract_data(filename)
                    data_from_file = (sim_num,rate_num, ratio, trace, score, var_fin, learn_time, n_selected, n_pruned)
                    accu.append(data_from_file)
                    print("{},{},{},{},{},{},{},{},{}".format(sim_num,rate_num, ratio, trace, score, var_fin, learn_time, n_selected, n_pruned))

df = pd.DataFrame(accu, columns=['count', 'rate_num', 'ratio', 'trace', 'score', 'var_fin', 'learn_time', 'n_selected', 'n_pruned'])
df.to_csv('aggregated_results_additional.csv', index=False)


# score, var_fin, learn_time, n_selected, n_pruned = extract_data('data0_rate1_ratio5_condTrue.pickle')
# print(score, var_fin, learn_time, n_selected, n_pruned)

