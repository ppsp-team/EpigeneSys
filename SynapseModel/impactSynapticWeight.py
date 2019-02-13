import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
from brian2 import *
prefs.codegen.target = 'numpy'  # use the Python fallback (cython bugs for some reason)

start_scope()

# Neuron parameters
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
taud = 20*ms # dopamine release time constant
taus = 1*ms # synaptic weight constant 
epsilon_dopa = 5e-1 # amount of dopamine released on reward


# usual neurons

neurons  = NeuronGroup(2, '''dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
                                      dge/dt = -ge / taue : 1''',
                      threshold='v>vt', reset='v = vr',
                      method='linear')
neurons.v = vr

# STDP synapse

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

synapse_stdp.connect(i = 0, j = 1)
synapse_stdp.mode = 0
# input 


indices =  [0]
times = [10*ms]
input = SpikeGeneratorGroup(1, indices, times)

# input synapse

input_synapse = Synapses(input, neurons, on_pre = '''v += 2000*mV''')
input_synapse.connect(i = 0, j = 0)

# parameters 
synapse_stdp.s = 0.005

# monitors
voltage_monitor = StateMonitor(neurons, "v", record = True)
conductance_monitor = StateMonitor(neurons, "ge", record = True)
synapse_weight = StateMonitor(synapse_stdp, "s", record = True)

# run the simulation
run(100*ms)
print(voltage_monitor.v[1].shape)

# plotting
figure(figsize = (9,9))
subplot(311)
plt.plot(voltage_monitor.t/ms, voltage_monitor.v[1])
subplot(312)
plt.plot(conductance_monitor.t/ms, conductance_monitor.ge[1])
subplot(313)
plt.plot(synapse_weight.t/ms, synapse_weight.s[0])

show()
