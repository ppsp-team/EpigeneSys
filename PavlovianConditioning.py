
#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
STDP modulated with reward

Adapted from Fig. 1c of:
Eugene M. Izhikevich 
Solving the distal reward problem through linkage of STDP and dopamine signaling. 
Cerebral cortex 17, no. 10 (2007): 2443-2452.

Note:
The variable "mode" can switch the behavior of the synapse from "Classical STDP" to "Dopamine modulated STDP".

Author: Guillaume Dumas (Institut Pasteur)
Date: 2018-08-24
'''
import matplotlib.pyplot as plt
from brian2 import *
start_scope()

"""Parameters"""
simulation_duration = 20 * second

# Neuron parameters
numberNeuronGroups = 25
neuronGroupSize = 50
taum = 10*ms
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms

# STDP parameters
taupre = 20*ms
taupost = taupre
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

# Dopamine signaling parameters
tauc = 1000*ms
taud = 200*ms
taus = 1*ms
epsilon_dopa = 5e-3

"""Initialize a network"""
network = Network()


"""Stimulus section"""

# Defining the stimuli
input_times = np.array([0] + [200*i + np.random.uniform(-50., 50.) for i in range(1,100)])*ms # define when the stimuli happen
input_indices = np.array([np.random.randint(0, numberNeuronGroups) for i in range(100)]) # define which group each pulse targets
groupedStimuli = SpikeGeneratorGroup(numberNeuronGroups,input_indices, input_times)
network.add(groupedStimuli)

noise_rate = 5*Hz
noise = PoissonGroup(numberNeuronGroups*neuronGroupSize, noise_rate)
network.add(noise)

# Defining the corresponding neurons
neurons = NeuronGroup(numberNeuronGroups*neuronGroupSize,  '''dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
                                      dge/dt = -ge / taue : 1''',
                      threshold='v>vt', reset='v = vr',
                      method='linear')
network.add(neurons)

inputSynapse = Synapses(groupedStimuli, neurons, model='''s : volt''', on_pre='v += s') # every time an input fires the corresponding neurons are depolarized beyond threshold
for k in range(numberNeuronGroups):
       inputSynapse.connect(i=[k], j=list(range(neuronGroupSize*k + 1, neuronGroupSize*(k+1)))) # input i is connected to all the neurons in the group Si
inputSynapse.s = 100*mV

noiseSynapse = Synapses(noise, neurons, model='''s : volt''', on_pre='v += s')
noiseSynapse.s = 100*mV
noiseSynapse.connect(condition = 'i=j', p = 1.)


# The synapse is ready to add !
network.add(inputSynapse)
network.add(noiseSynapse)


"""STDP section"""
epsilon = 0.1 # sparseness of synaptic connections

synapse_stdp = Synapses(neurons, neurons,
                   model='''mode: 1
                         dc/dt = -c / tauc : 1 (clock-driven)
                         dd/dt = -d / taud : 1 (clock-driven)
                         ds/dt = mode * c * d / taus : 1 (clock-driven)
                         dApre/dt = -Apre / taupre : 1 (event-driven)
                         dApost/dt = -Apost / taupost : 1 (event-driven)''',
                   on_pre='''ge += s
                          Apre += dApre
                          c = clip(c + mode * Apost, -gmax, gmax)
                          s = clip(s + (1-mode) * Apost, -gmax, gmax)
                          ''',
                   on_post='''Apost += dApost
                          c = clip(c + mode * Apre, -gmax, gmax)
                          s = clip(s + (1-mode) * Apre, -gmax, gmax)
                          ''',
                   method='euler'
                   )


# Connectivity: make sure a neuron does not form a synapse with itself
synapse_stdp.connect(condition = 'i!=j', p=epsilon)

# Initialize the parameters of the synapses
synapse_stdp.mode = 0
synapse_stdp.s = 1e-10
synapse_stdp.c = 1e-10
synapse_stdp.d = 0

# The synapse is ready to add !
network.add(synapse_stdp)

"""Dopamine signaling section"""

dopamine = NeuronGroup(1, '''v : volt''', threshold='v>1*volt', reset='v=0*volt')
network.add(dopamine)

# To monitor when dopamine is released
dopamine_monitor = SpikeMonitor(dopamine)
network.add(dopamine_monitor)


# Synapse ordering dopamine release when the conditioning stimulus occurs
conditioning = Synapses(groupedStimuli, dopamine, on_pre='''v_post += 2*volt''', method='exact') # 2*volt is bigger than the dopamine neuron's 1*volt threshold
conditioning.connect(i  = 0, j = 0)
network.add(conditioning)

# Synapse that accounts for the effect of dopamine on plasticity
reward = Synapses(dopamine, synapse_stdp, on_pre='''d_post += epsilon_dopa''', method='exact')
reward.connect(p=1.) # every synapse is affected by dopamine
reward.delay='100*ms' # dopamine reaches the synapse 100ms after the conditioning stimulus

# The synapse is ready to add !
network.add(reward)

# Simulation
## Classical STDP
#synapse_stdp.mode = 0

## Dopamine modulated STDP
synapse_stdp.mode = 1
neuronSpikes = SpikeMonitor(neurons, record=True)
network.add(neuronSpikes)
network.run(simulation_duration, report='text')


plt.plot(neuronSpikes.t/ms, neuronSpikes.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()
