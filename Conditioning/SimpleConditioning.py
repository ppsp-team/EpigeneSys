import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import pickle
import argparse
import os
from brian2 import *
from affichage import affichage

# Parsing arguments

parser = argparse.ArgumentParser(description='Run simple conditioning')
parser.add_argument('-n', '--numberNeuronGroups', type=int, default=10,
                   help="number of neuron groups in the simulation")
parser.add_argument('-s', '--neuronGroupSize', type=int, default=10, 
                    help="number of neurons in one group")
parser.add_argument('-t', '--time', type=int, default=500,
                   help='duration of the simulation')
parser.add_argument('-d', '--dir', type=str, default='results',
                   help='directory for storing results')
parser.add_argument('--delay', type=int, default='100',
                    help='delay of dopamine in ms')
parser.add_argument('-o', '--output', type=str, default='',
                   help='name of output file')
parser.add_argument('-c', '--cpp_standalone', action='store_true', default=False,
                   help='run with cpp standalone code generation')

args = parser.parse_args()


# Parameters

results_dir = args.dir
if args.output == '':
    args.output = "output_file_{}x{}_{}s_{}ms".format(args.numberNeuronGroups, args.neuronGroupSize, args.time, args.delay)
output_file = os.path.normpath(os.path.join(results_dir, args.output))

# if output file already exists :
n = 2
while os.path.exists(output_file):
    if output_file.endswith("({})".format(str(n-1))):
        output_file = output_file[:-3] + "({})".format(str(n))
    else:
        output_file += "({})".format(str(n))
    n += 1

with open(output_file, 'wb') as file:
    pass

if args.cpp_standalone:
    set_device('cpp_standalone')
else:
    prefs.codegen.target = 'cython'

numberNeuronGroups = args.numberNeuronGroups
neuronGroupSize = args.neuronGroupSize
simulation_duration = args.time * second

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
if args.cpp_standalone:
    synapses_monitor = StateMonitor(synapse_stdp, ['s'], record=list(range(int(len(neurons)**2*epsilon))), dt=synaptic_weight_interval)
    # Necessary for cpp standalone code. It's not convenient but brian does not handle record=True with the generated code.
else:
    synapses_monitor = StateMonitor(synapse_stdp, ['s'], record=True, dt=synaptic_weight_interval)
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
reward.delay='{}*ms'.format(args.delay) # dopamine reaches the synapse 100ms after the conditioning stimulus

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
output['time'] = synapses_monitor.t/ms
output['synaptic_weights'] = synapses_monitor.s
output['idx'] = np.array(synapse_stdp.i)
output['spike_t'] = neuronSpikes.t/ms
output['spike_i'] = np.array(neuronSpikes.i)
output['dopa'] = dopamine_monitor.t/ms

with open(output_file, 'wb') as file:
    pickle.dump(output, file)

affichage(output_file, numberNeuronGroups, neuronGroupSize, args.time)