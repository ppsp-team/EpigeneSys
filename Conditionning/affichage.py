import numpy as np
import pickle
import matplotlib.pyplot as plt

results_dir = "C:\\Users\\Valentin\\Documents\\Projets\\PSC\\Resultats\\Conditionning\\Data\\"

# Unpickling data from output file

with open (results_dir + 'output_file', 'rb') as file:
    output = pickle.load(file)
    
t = output['t']
s = output['s']
i = output['i']
spike_t = output['spike_t']
spike_i = output['spike_i']
dopa = output['dopa']

# Cleaning the data

s = s[:min(len(i), len(s))]
i = i[:len(s)]

# Printing results

plt.plot(t, s[i < 10].mean(axis=0), 'g', label='group 1')
plt.plot(t, s[i < 10].mean(axis=0)+np.sqrt(s[i < 10].var(axis=0)), 'g--', label='group1 + standard deviation')
plt.plot(t, s[i < 10].mean(axis=0)-np.sqrt(s[i < 10].var(axis=0)), 'g--', label='group1 - standard deviation')
plt.plot(t, s.mean(axis=0), 'b', label='mean')
plt.plot(t, s.mean(axis=0)+np.sqrt(s.var(axis=0)), 'b--', label='mean + standard deviation')
plt.ylabel('Average synaptic weight')
plt.xlabel('Time (ms)')
plt.legend()
plt.tight_layout()
plt.savefig(results_dir + 'output.pdf', format='pdf')
plt.show()