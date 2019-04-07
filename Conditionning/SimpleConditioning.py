import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import pickle
from brian2 import *

"""Parameters"""
memory_intensive = False
results_dir = "C:\\Users\\Valentin\\Documents\\Projets\\PSC\\Resultats\\Conditionning\\Data\\"

if memory_intensive:
    numberNeuronGroups = 10
    neuronGroupSize = 10
    simulation_duration = 500 * second
    synaptic_weight_interval = 1 * second
    prefs.codegen.target = 'cython'
    
else:
    set_device('cpp_standalone')
    numberNeuronGroups = 2
    neuronGroupSize = 5
    simulation_duration = 120 * second
    synaptic_weight_interval = 20 * second

start_scope()

# Neuron parameters
taum = 1*ms # neuron equation time constant
Ee = 0*mV # excitatory synapse equilibrium potential
vt = -54*mV # threshold potential
vr = -70*mV # resting potential
El = -70*mV # leak channel potential
taue = 0.5*ms # synaptic conductance time constant

# STDP
taupre = 20*ms
taupost = taupre
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost
dApost *= gmax
dApre *= gmax

# Dopamine signaling
tauc = 1000*ms # eligibility time constant
taud = 200*ms # dopamine release time constant
taus = 1*ms # synaptic weight constant 
epsilon_dopa = 5e-3 # amount of dopamine released on reward


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


# The synapses are ready to add !
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
                   on_pre='''ge += 40*s
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

# To monitor the synaptic weight
if memory_intensive:
    synapses_monitor = StateMonitor(synapse_stdp, ['s'], record=True, dt=synaptic_weight_interval)
else:
    synapses_monitor = StateMonitor(synapse_stdp, ['s'], record=list(range(int(len(neurons)**2*epsilon))), dt=synaptic_weight_interval)
    # Necessary for cpp standalone code. It's not convenient but brian does not handle record=True with the generated code.
network.add(synapses_monitor)

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

# Dopamine modulated STDP
synapse_stdp.mode = 1
neuronSpikes = SpikeMonitor(neurons, record=list(range(len(neurons))))
network.add(neuronSpikes)

# Running the network
network.run(simulation_duration, report='text')

# Results

output = dict()
output['t'] = synapses_monitor.t/ms
output['s'] = synapses_monitor.s
output['i'] = np.array(synapse_stdp.i)
output['spike_t'] = neuronSpikes.t/ms
output['spike_i'] = np.array(neuronSpikes.i)
output['dopa'] = dopamine_monitor.t/ms

try :
    with open(results_dir + 'output_file', 'wb') as file:
        pickle.dump(output, file)
except:
    with open('output_file', 'wb') as file:
        pickle.dump(output, file)


#figure(figsize=(9,18))
#subplot(311)
#rewardTimes = [t for t in dopamine_monitor.t/ms if t < 5000]
#if len(rewardTimes) > 0:
#   plt.axvline(x = rewardTimes[0], linestyle = '-', color = 'orange', label='dopamine release')
#for rewardTime in rewardTimes[1:]:
#   plt.axvline(x = rewardTime, linestyle = '-', color = 'orange')
#spikeTimes = [t for t in neuronSpikes.t/ms if t < 5000]
#spikeIndex = neuronSpikes.i[:len(spikeTimes)]
#plt.plot(spikeTimes, spikeIndex, '.', markersize=3, label='neuron spike')
#ylabel('Neuron Index')
#xlabel('Time (ms)')
#plt.legend()

#subplot(312)
#rewardTimes = [t for t in dopamine_monitor.t/ms if t > simulation_duration*1000/second-5000]
#if len(rewardTimes) > 0:
#   plt.axvline(x = rewardTimes[0], linestyle = '-', color = 'orange', label='dopamine release')
#for rewardTime in rewardTimes[1:]:
#   plt.axvline(x = rewardTime, linestyle = '-', color = 'orange')
#spikeTimes = [t for t in neuronSpikes.t/ms if t > simulation_duration*1000/second-5000]
#spikeIndex = neuronSpikes.i[-len(spikeTimes):]
#plt.plot(spikeTimes, spikeIndex, '.', markersize=3, label='neuron spike')
#ylabel('Neuron Index')
#xlabel('Time (ms)')
#plt.legend()

#subplot(313)
#group1 = np.array([0.] * len(synapses_monitor.t))
#group1_nb_synapses = 0
#mean = np.array([0.] * len(synapses_monitor.t))
#mean_nb_synapses = 0
#other = np.array([0.] * len(synapses_monitor.t))
#other_nb_synapses = 0
#for i in range(len(synapses_monitor.s)):
#    if synapse_stdp.i[i] < neuronGroupSize:
#        group1 += synapses_monitor.s[i]
#        group1_nb_synapses += 1
#    else:
#        other += synapses_monitor.s[i]
#        other_nb_synapses += 1
#    mean += synapses_monitor.s[i]
#    mean_nb_synapses += 1
#mean = mean / mean_nb_synapses
#group1 = group1/group1_nb_synapses
#other = other/other_nb_synapses
#plt.plot(synapses_monitor.t, group1, label='group 1')
#plt.plot(synapses_monitor.t, other, label='other')
#plt.plot(synapses_monitor.t, mean, label='mean')
#ylabel('Average synaptic weight')
#xlabel('Time (s)')
#plt.legend()
#tight_layout()
#show(block=True)
