
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
from brian2 import *
prefs.codegen.target = 'numpy'  # use the Python fallback (cython bugs for some reason)

start_scope()

"""Parameters"""
simulation_duration = 1000 * second

# Neuron parameters
numberNeuronGroups = 2
neuronGroupSize = 50
taum = 1*ms # neuron equation time constant
Ee = 0*mV # excitatory synapse equilibrium potential
vt = -54*mV # threshold potential
vr = -70*mV # resting potential
El = -70*mV # leak channel potential
taue = 0.5*ms # synaptic conductance time constant

## STDP
taupre = 20*ms
taupost = taupre
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

## Dopamine signaling
tauc = 1000*ms # eligibility time constant
taud = 200*ms # dopamine release time constant
taus = 1*ms # synaptic weight constant 
epsilon_dopa = 5e-1 # amount of dopamine released on reward


"""Initialize a network"""
network = Network()


"""Stimulus section"""

# Defining the stimuli
numberStimulations = int((1000*simulation_duration/second - 100)/200) # stimulus every ~200 seconds
input_times = np.array([100 + 200*i + np.random.randint(-50, 50) for i in range(numberStimulations)])*ms # define when the stimuli happen 
input_indices = np.array([np.random.randint(0, numberNeuronGroups) for i in range(numberStimulations)]) # define which group each pulse targets
groupedStimuli = SpikeGeneratorGroup(numberNeuronGroups,input_indices, input_times)
network.add(groupedStimuli)

# Defining the noise

noise_rate = 2*Hz
noise = PoissonGroup(numberNeuronGroups*neuronGroupSize, noise_rate)
network.add(noise)

# Defining the neurons we are actually interested in
neurons = NeuronGroup(numberNeuronGroups*neuronGroupSize,  '''dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
                                      dge/dt = -ge / taue : 1''',
                      threshold='v>vt', reset='v = vr',
                      method='linear')
neurons.v = vr
network.add(neurons)

# Synapse between the stimuli and our neurons

inputSynapse = Synapses(groupedStimuli, neurons, model='''s : volt''', on_pre='v += s') # every time an input fires the corresponding neurons are depolarized beyond threshold
for k in range(numberNeuronGroups):
       inputSynapse.connect(i=[k], j=list(range(neuronGroupSize*k + 1, neuronGroupSize*(k+1)))) # input i is connected to all the neurons in the group Si
inputSynapse.s = 100*mV # enough to guarantee a postsynaptic spike


# Synapse between the noise and our neurons

noiseSynapse = Synapses(noise, neurons, model='''s : volt''', on_pre='v += s')
noiseSynapse.connect(condition = 'i==j', p = 1.)
noiseSynapse.s = 100*mV # enough to guarantee a postsynaptic spike


 #The synapses are ready to add !
network.add(inputSynapse)
network.add(noiseSynapse)


"""STDP section"""
epsilon = 0.5 # sparseness of synaptic connections

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
                     method='''euler'''
                   )


# Connectivity: make sure a neuron does not form a synapse with itself
synapse_stdp.connect(condition = 'i!=j', p=epsilon)

# Initialize the parameters of the synapses
synapse_stdp.mode = 1
synapse_stdp.s = 1e-9
synapse_stdp.c = 1e-9
synapse_stdp.d = 0

# The synapse is ready to add !
network.add(synapse_stdp)

"""Dopamine signaling section"""

dopamine = NeuronGroup(1, '''v : volt''', threshold='v>1*volt', reset='v=0*volt')
network.add(dopamine)


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
synapse_stdp.mode = 0
neuronSpikes = SpikeMonitor(neurons, record=True)
network.add(neuronSpikes)
network.run(simulation_duration, report='text')


for xStim in input_times/ms:
       plt.axvline(x = xStim, linestyle = '-', color = 'orange')

plt.plot(neuronSpikes.t/ms, neuronSpikes.i, '.', markersize=3)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')

plt.show()


tab = []
for t0 in input_times/ms:
    tab.append(len([t for t in neuronSpikes.t/ms if t>= t0 and t < t0+50]))
tab