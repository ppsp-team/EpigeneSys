import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt

def affichage(output_file, number=10, size=10, duration=500):

    # Unpickling data from output file

    with open (output_file, 'rb') as file:
        output = pickle.load(file)
    
    t = output['time']
    synaptic_weights = output['synaptic_weights']
    idx = output['idx']
    spike_t = output['spike_t']
    spike_i = output['spike_i']
    dopa = output['dopa']

    # Cleaning the data

    synaptic_weights = synaptic_weights[:min(len(idx), len(synaptic_weights))]
    idx = idx[:len(synaptic_weights)]

    # Printing results

    plt.plot(t, synaptic_weights[idx < size].mean(axis=0), 'g', label='group 1')
    plt.plot(t, synaptic_weights[idx < size].mean(axis=0)+np.sqrt(synaptic_weights[idx < size].var(axis=0)/size), 'g--', label='group1 + standard error')
    plt.plot(t, synaptic_weights[idx < size].mean(axis=0)-np.sqrt(synaptic_weights[idx < size].var(axis=0)/size), 'g--', label='group1 - standard erro')
    plt.plot(t, synaptic_weights.mean(axis=0), 'b', label='mean')
    plt.plot(t, synaptic_weights.mean(axis=0)+np.sqrt(synaptic_weights.var(axis=0)/(number*size)), 'b--', label='mean + standard error')

    end_results = list()
    max = 0
    i_max = 0
    for i in range(number):
        end_results.append(synaptic_weights[(idx < number*i + size) & (idx >= number*i)][:, -1].mean(axis=0))
        if end_results[-1] > max:
            max = end_results[-1]
            i_max = i
    
    plt.plot(t, synaptic_weights[(idx < number*i_max + size) & (idx >= number*i_max)].mean(axis=0), 'r--', label='group max')

    plt.ylabel('Average synaptic weight')
    plt.xlabel('Time (ms)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file + ".pdf", format='pdf')
