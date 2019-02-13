from brian2 import *
import numpy as np

# Parameters
simulationDuration = 1200 * second
numberNeuronsKernel = 100

## Neurons
taum = 10*ms
taue = 5*ms
vt = -54*mV
vr = -60*mV
El = -60*mV
Ee = 0*mV
gmax = 100 # Maximum inhibitory weight

## STDP
taupre = 20*ms
taupost = taupre
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

## Dopamine signaling
tauc = 1000*ms
taud = 200*ms
taus = 1*ms
epsilon_dopa = 5e-3

## Dopamine monitor
taumonitor = 1 * second
monitorIncrement = 0.4

# Setting the stage
network = Network()

indicesInput1 = list()
indicesInput2 = list()

for k in range(int(simulationDuration/(10*second))):
    bit = np.random.randint(0, 2)
    indicesInput1 += [bit] * 1000
    bit = np.random.randint(0, 2)
    indicesInput2 += [bit] * 1000

times = [k*10*ms for k in range(int(simulationDuration/(10*ms)))]


"""NEURON GROUPS"""

# Inputs: two neurons, one for 0, one for 1 
input1Generator = SpikeGeneratorGroup(2, indicesInput1, times)
network.add(input1Generator)
input2Generator = SpikeGeneratorGroup(2, indicesInput2, times)
network.add(input2Generator)

input1 = NeuronGroup(2, 
                    '''dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
                                      dge/dt = -ge / taue : 1''',
                    threshold='v>vt', 
                    reset='v = vr',
                    method='euler')
input1.v = vr
network.add(input1)

input2 = NeuronGroup(2, 
                    '''dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
                                      dge/dt = -ge / taue : 1''',
                    threshold='v>vt', 
                    reset='v = vr',
                    method='euler')
input2.v = vr
network.add(input2)




# Kernel: for now, all excitatory, do distinction between cortical layers
kernel = NeuronGroup(numberNeuronsKernel, 
                    '''dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
                                      dge/dt = -ge / taue : 1''',
                    threshold='v>vt', 
                    reset='v = vr',
                    method='euler')
kernel.v = vr
network.add(kernel)

kernelMonitor = SpikeMonitor(kernel)
network.add(kernelMonitor)


# Ouput: for now, all excitatory, no distinction between cortical layers
output = NeuronGroup(2, 
                   '''dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
                                      dge/dt = -ge / taue : 1''',
                    threshold='v>vt', 
                    reset='v = vr',
                    method='euler')
output.v = vr
network.add(output)

outputMonitor = SpikeMonitor(output, record=True)
network.add(outputMonitor)


"""SYNAPSES"""

# Synapse from the spike generator group to the actual input neurons
input1Synapse = Synapses(input1Generator, input1, on_pre = 'v += vt-El')
input1Synapse.connect(condition = 'i == j')
network.add(input1Synapse)

input2Synapse = Synapses(input2Generator, input2, on_pre = 'v += vt-El')
input2Synapse.connect(condition = 'i == j')
network.add(input2Synapse)

# Kernel synapse 
kernelDensity = 0.1
kernelSynapses = Synapses(kernel, kernel, 
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
                      method='euler')
kernelSynapses.connect(p = kernelDensity)

kernelSynapses.mode = 1
kernelSynapses.s = 0.01
kernelSynapses.c = 0.01
kernelSynapses.d = 0

network.add(kernelSynapses)


# Inputs to kernel
input1ToKernel = Synapses(input1, kernel,
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
                      method='euler')

input1ToKernel.connect(p = 0.2)
input1ToKernel.mode = 1
input1ToKernel.s = 0.01
input1ToKernel.c = 0.01
input1ToKernel.d = 0

network.add(input1ToKernel)

input2ToKernel = Synapses(input2, kernel,
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
                      method='euler')

input2ToKernel.connect()
input2ToKernel.mode = 1
input2ToKernel.s = 0.01
input2ToKernel.c = 0.01
input2ToKernel.d = 0

network.add(input2ToKernel)

# Kernel to output

kernelToOutput = Synapses(kernel, output,
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
                      method='euler')
kernelToOutput.connect()
kernelToOutput.mode = 1
kernelToOutput.s = 0.01
kernelToOutput.c = 0.01
kernelToOutput.d = 0

network.add(kernelToOutput)

# Dopamine monitor

dopamineMonitor = NeuronGroup(1, '''
                                   mode : 1
                                   dmonitor/dt = monitor / taumonitor : 1
                                   ''', # mode = 1 when xor answer is 1 and -1 when it is 0
                                   threshold='monitor > 1',
                                   reset='monitor=0',
                                   method='linear')
dopamineMonitor.mode = 0
network.add(dopamineMonitor)

indicesModeChanger = [int(in1 != in2) for in1, in2 in zip(indicesInput1, indicesInput2)]
dopamineMonitorModeChanger = SpikeGeneratorGroup(2, indicesModeChanger, times)
network.add(dopamineMonitorModeChanger)

dopamineMonitorModeChangerSynapse0 = Synapses(dopamineMonitorModeChanger, dopamineMonitor, on_pre='mode = 1')
dopamineMonitorModeChangerSynapse1 = Synapses(dopamineMonitorModeChanger, dopamineMonitor, on_pre='mode = -1')
dopamineMonitorModeChangerSynapse0.connect(i=0, j=0)
dopamineMonitorModeChangerSynapse1.connect(i=1, j=0)
network.add(dopamineMonitorModeChangerSynapse0)
network.add(dopamineMonitorModeChangerSynapse1)


output0ToMonitorSynapse = Synapses(output[:1], dopamineMonitor, on_pre='monitor += (-mode)*monitorIncrement')
output1ToMonitorSynapse = Synapses(output[1:], dopamineMonitor, on_pre='monitor += mode*monitorIncrement')
output0ToMonitorSynapse.connect(p=1.)
output1ToMonitorSynapse.connect(p=1.)
network.add(output0ToMonitorSynapse)
network.add(output1ToMonitorSynapse)

dopamineDispenser1 = Synapses(dopamineMonitor, kernelSynapses, on_pre='d += epsilon_dopa', method='exact')
dopamineDispenser2 = Synapses(dopamineMonitor, input1ToKernel, on_pre='d += epsilon_dopa', method='exact')
dopamineDispenser3 = Synapses(dopamineMonitor, input2ToKernel, on_pre='d += epsilon_dopa', method='exact')
dopamineDispenser4 = Synapses(dopamineMonitor, kernelToOutput, on_pre='d += epsilon_dopa', method='exact')
dopamineDispenser5 = Synapses(dopamineMonitor, kernelToOutput, on_pre='d += epsilon_dopa', method='exact')
dopamineDispenser1.connect(p=1.)
dopamineDispenser2.connect(p=1.)
dopamineDispenser3.connect(p=1.)
dopamineDispenser4.connect(p=1.)
dopamineDispenser5.connect(p=1.)
network.add(dopamineDispenser1)
network.add(dopamineDispenser2)
network.add(dopamineDispenser3)
network.add(dopamineDispenser4)
network.add(dopamineDispenser5)



'''What does the network look like?'''
'''
import matplotlib.pyplot as plt
import numpy.random as random
import matplotlib.patches as mpatches

plt.scatter([0,1], [10, 10], c = 'r', s = 100) # input1Generator
plt.scatter([9,10], [10, 10], c = 'r', s = 100) # input2Generator
plt.scatter([0,1], [8, 8], c = 'b', s = 100) # input1
plt.scatter([9,10], [8, 8], c = 'b', s = 100) # input2
plt.scatter(random.randn(numberNeuronsKernel) + 5 , random.randn(numberNeuronsKernel) + 4, c = 'g', s = 10) #kernel
plt.scatter([4.5,5.5], [0,0], c = 'orange', s = 100) # input1

red_patch = mpatches.Patch(color='r', label='Input generators')
blue_patch = mpatches.Patch(color='b', label='Inputs')
green_patch = mpatches.Patch(color='g', label='Kernel')
orange_patch = mpatches.Patch(color='orange', label='Output')
plt.legend(handles=[red_patch,blue_patch,green_patch,orange_patch])


plt.show()
'''
network.run(10*second)
plt.plot(kernelMonitor.t/ms, kernelMonitor.i, '.')
plt.show()

plt.plot(outputMonitor.t/ms, outputMonitor.i, '.')
plt.show()