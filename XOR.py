from brian2 import *
import numpy as np

# Parameters
simulationDuration = 1200 * second
numberNeuronsKernel = 100
eta = 0.1 # learning rate

## Neurons
vt = -54*mV
vr = -60*mV
gl = 10.0*nsiemens # leak channel
el = -60*mV
gmax = 100 # Maximum inhibitory weight
tau_ampa = 5.0*ms   # Glutamatergic synaptic time constant 
tau_gaba = 10.0*ms  # GABAergic synaptic time constant
tau_stdp = 20*ms    # STDP time constant
alpha_s = 4*Hz*tau_stdp*2  # Target rate parameter
memc = 200*pfarad

## STDP
taupre = 20*ms
taupost = taupre
gmax = 100
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



'''Simplest setup: bits encoded as bits, not inferred from the activity in an assembly of neurons'''
# Inputs: two neurons, one for 0, one for 1 
input1Generator = SpikeGeneratorGroup(2, indicesInput1, times)
network.add(input1Generator)
input2Generator = SpikeGeneratorGroup(2, indicesInput2, times)
network.add(input2Generator)

input1 = NeuronGroup(numberNeuronsKernel, 
                    ''' dv/dt=(-gl*(v-el)-(gAmpa*v+gGaba*(v-vr)))/memc : volt (unless refractory)
                        dgAmpa/dt = -gAmpa/tau_ampa : siemens
                        dgGaba/dt = -gGaba/tau_gaba : siemens''',
                    threshold='v>vt', 
                    reset='v = vr',
                    refractory=5*ms,
                    method='euler')
network.add(input1)

input2 = NeuronGroup(numberNeuronsKernel, 
                    ''' dv/dt=(-gl*(v-el)-(gAmpa*v+gGaba*(v-vr)))/memc : volt (unless refractory)
                        dgAmpa/dt = -gAmpa/tau_ampa : siemens
                        dgGaba/dt = -gGaba/tau_gaba : siemens''',
                    threshold='v>vt', 
                    reset='v = vr',
                    refractory=5*ms,
                    method='euler')
network.add(input2)




# Kernel: for now, all excitatory, do distinction between cortical layers
kernel = NeuronGroup(numberNeuronsKernel, 
                    ''' dv/dt=(-gl*(v-el)-(gAmpa*v+gGaba*(v-vr)))/memc : volt (unless refractory)
                        dgAmpa/dt = -gAmpa/tau_ampa : siemens
                        dgGaba/dt = -gGaba/tau_gaba : siemens''',
                    threshold='v>vt', 
                    reset='v = vr',
                    refractory=5*ms,
                    method='euler')
network.add(kernel)

percentInh = 0.2
percentExc = 1 - percentInh

# Ouput: for now, all excitatory, do distinction between cortical layers
output = NeuronGroup(2, 
                    ''' dv/dt=(-gl*(v-el)-(gAmpa*v+gGaba*(v-vr)))/memc : volt (unless refractory)
                        dgAmpa/dt = -gAmpa/tau_ampa : siemens
                        dgGaba/dt = -gGaba/tau_gaba : siemens''',
                    threshold='v>vt', 
                    reset='v = vr',
                    refractory=5*ms,
                    method='euler')
network.add(output)

# Synapse from the spike generator group to the actual input neurons
input1Synapse = Synapses(input1Generator, input1, on_pre = 'v += vt-el')
input2Synapse = Synapses(input2Generator, input1, on_pre = 'v += vt-el')
input1Synapse.connect()
input2Synapse.connect()
network.add(input1Synapse)
network.add(input2Synapse)

# Excitatory synapse in kernel connected to everyone 
excToAllDensity = 0.1
synapseExcToAll = Synapses(kernel[:int(percentExc*numberNeuronsKernel)], kernel, 
                      model='''mode: 1
                         dc/dt = -c / tauc : 1 (clock-driven)
                         dd/dt = -d / taud : 1 (clock-driven)
                         ds/dt = mode * c * d / taus : 1 (clock-driven)
                         dApre/dt = -Apre / taupre : 1 (event-driven)
                         dApost/dt = -Apost / taupost : 1 (event-driven)''',
                      on_pre='''gAmpa += s*nS
                          Apre += dApre
                          c = clip(c + mode * Apost, -gmax, gmax)
                          s = clip(s + (1-mode) * Apost, -gmax, gmax)
                          ''',
                      on_post='''Apost += dApost
                          c = clip(c + mode * Apre, -gmax, gmax)
                          s = clip(s + (1-mode) * Apre, -gmax, gmax)
                          ''',
                      method='euler')
synapseExcToAll.connect(p = excToAllDensity)
network.add(synapseExcToAll)

# Inhibitory synapses in kernel to inhibitory neurons
inhToInhDensity = 0.1
synapseInhToInh = Synapses(kernel[int(percentExc*numberNeuronsKernel)+1:numberNeuronsKernel], kernel[int(percentExc*numberNeuronsKernel)+1:numberNeuronsKernel], 
                      model='''mode: 1
                         dc/dt = -c / tauc : 1 (clock-driven)
                         dd/dt = -d / taud : 1 (clock-driven)
                         ds/dt = mode * c * d / taus : 1 (clock-driven)
                         dApre/dt = -Apre / taupre : 1 (event-driven)
                         dApost/dt = -Apost / taupost : 1 (event-driven)''',
                      on_pre='''gGaba += 10*s*nS
                          Apre += dApre
                          c = clip(c + mode * Apost, -gmax, gmax)
                          s = clip(s + (1-mode) * Apost, -gmax, gmax)
                          ''', # There is an arbitrary times 10 factor 
                      on_post='''Apost += dApost
                          c = clip(c + mode * Apre, -gmax, gmax)
                          s = clip(s + (1-mode) * Apre, -gmax, gmax)
                          ''',
                      method='euler')
synapseInhToInh.connect(p = inhToInhDensity)
network.add(synapseInhToInh)

# Inhibitory synapses in kernel to excitatory neurons: this is where the plasticity happens
inhToExcDensity = 0.1
synapseInhToExc = Synapses(kernel[int(percentExc*numberNeuronsKernel)+1:numberNeuronsKernel], kernel[:int(percentExc*numberNeuronsKernel)],   
                      model='''mode: 1
                         dc/dt = -c / tauc : 1 (clock-driven)
                         dd/dt = -d / taud : 1 (clock-driven)
                         ds/dt = mode * c * d / taus : 1 (clock-driven)
                         dApre/dt = -Apre / taupre : 1 (event-driven)
                         dApost/dt = -Apost / taupost : 1 (event-driven)''',
                      on_pre='''gGaba += 10*s*nS
                          Apre += dApre
                          c = clip(c + mode * Apost, -gmax, gmax)
                          s = clip(s + (1-mode) * Apost, -gmax, gmax)
                          ''', # There is an arbitrary times 10 factor 
                      on_post='''Apost += dApost
                          c = clip(c + mode * Apre, -gmax, gmax)
                          s = clip(s + (1-mode) * Apre, -gmax, gmax)
                          ''',
                      method='euler')
synapseInhToExc.connect(p = inhToExcDensity)
network.add(synapseInhToExc)


# Synapses between inputs and kernel: only connect to the excitatory neurons in the kernel
input1KernelSynapses = Synapses(input1, kernel[:int(percentExc*numberNeuronsKernel)],
                     model='''mode: 1
                         dc/dt = -c / tauc : 1 (clock-driven)
                         dd/dt = -d / taud : 1 (clock-driven)
                         ds/dt = mode * c * d / taus : 1 (clock-driven)
                         dApre/dt = -Apre / taupre : 1 (event-driven)
                         dApost/dt = -Apost / taupost : 1 (event-driven)''',
                      on_pre='''gAmpa += s*nS
                          Apre += dApre
                          c = clip(c + mode * Apost, -gmax, gmax)
                          s = clip(s + (1-mode) * Apost, -gmax, gmax)
                          ''',
                      on_post='''Apost += dApost
                          c = clip(c + mode * Apre, -gmax, gmax)
                          s = clip(s + (1-mode) * Apre, -gmax, gmax)
                          ''',
                      method='euler')

input2KernelSynapses = Synapses(input2, kernel[:int(percentExc*numberNeuronsKernel)],   
                      model='''mode: 1
                         dc/dt = -c / tauc : 1 (clock-driven)
                         dd/dt = -d / taud : 1 (clock-driven)
                         ds/dt = mode * c * d / taus : 1 (clock-driven)
                         dApre/dt = -Apre / taupre : 1 (event-driven)
                         dApost/dt = -Apost / taupost : 1 (event-driven)''',
                      on_pre='''gAmpa += s*nS
                          Apre += dApre
                          c = clip(c + mode * Apost, -gmax, gmax)
                          s = clip(s + (1-mode) * Apost, -gmax, gmax)
                          ''',
                      on_post='''Apost += dApost
                          c = clip(c + mode * Apre, -gmax, gmax)
                          s = clip(s + (1-mode) * Apre, -gmax, gmax)
                          ''',
                      method='euler')
input1KernelSynapses.connect()
input2KernelSynapses.connect()
network.add(input1KernelSynapses)
network.add(input2KernelSynapses)

# Synapses between kernel and output

kernelExcToOutputSynapses = Synapses(kernel, output,
                      model='''mode: 1
                         dc/dt = -c / tauc : 1 (clock-driven)
                         dd/dt = -d / taud : 1 (clock-driven)
                         ds/dt = mode * c * d / taus : 1 (clock-driven)
                         dApre/dt = -Apre / taupre : 1 (event-driven)
                         dApost/dt = -Apost / taupost : 1 (event-driven)''',
                      on_pre='''gAmpa += s*nS
                          Apre += dApre
                          c = clip(c + mode * Apost, -gmax, gmax)
                          s = clip(s + (1-mode) * Apost, -gmax, gmax)
                          ''',
                      on_post='''Apost += dApost
                          c = clip(c + mode * Apre, -gmax, gmax)
                          s = clip(s + (1-mode) * Apre, -gmax, gmax)
                          ''',
                      method='euler')
kernelInhToOutputSynapses = Synapses(kernel, output,
                      model='''mode: 1
                         dc/dt = -c / tauc : 1 (clock-driven)
                         dd/dt = -d / taud : 1 (clock-driven)
                         ds/dt = mode * c * d / taus : 1 (clock-driven)
                         dApre/dt = -Apre / taupre : 1 (event-driven)
                         dApost/dt = -Apost / taupost : 1 (event-driven)''',
                      on_pre='''gGaba += 10*s*nS
                          Apre += dApre
                          c = clip(c + mode * Apost, -gmax, gmax)
                          s = clip(s + (1-mode) * Apost, -gmax, gmax)
                          ''', # There is an arbitrary times 10 factor
                      on_post='''Apost += dApost
                          c = clip(c + mode * Apre, -gmax, gmax)
                          s = clip(s + (1-mode) * Apre, -gmax, gmax)
                          ''',
                      method='euler')

kernelExcToOutputSynapses.connect()
kernelInhToOutputSynapses.connect()
network.add(kernelExcToOutputSynapses)
network.add(kernelInhToOutputSynapses)


# Dopamine monitor

dopamineMonitor = NeuronGroup(1, '''
                                   mode : 1
                                   dmonitor/dt = monitor / taumonitor : 1
                                   ''', # mode = 1 when xor answer is 1 and -1 when it is 0
                                   threshold='monitor > 1',
                                   reset='montior=0',
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

dopamineDispenser1 = Synapses(dopamineMonitor, synapseExcToAll, on_pre='d += epsilon_dopa', method='exact')
dopamineDispenser2 = Synapses(dopamineMonitor, synapseInhToInh, on_pre='d += epsilon_dopa', method='exact')
dopamineDispenser3 = Synapses(dopamineMonitor, synapseInhToExc, on_pre='d += epsilon_dopa', method='exact')
dopamineDispenser4 = Synapses(dopamineMonitor, input1KernelSynapses, on_pre='d += epsilon_dopa', method='exact')
dopamineDispenser5 = Synapses(dopamineMonitor, input2KernelSynapses, on_pre='d += epsilon_dopa', method='exact')
dopamineDispenser6 = Synapses(dopamineMonitor, kernelExcToOutputSynapses, on_pre='d += epsilon_dopa', method='exact')
dopamineDispenser7 = Synapses(dopamineMonitor, kernelInhToOutputSynapses, on_pre='d += epsilon_dopa', method='exact')
dopamineDispenser1.connect(p=1.)
dopamineDispenser2.connect(p=1.)
dopamineDispenser3.connect(p=1.)
dopamineDispenser4.connect(p=1.)
dopamineDispenser5.connect(p=1.)
dopamineDispenser6.connect(p=1.)
dopamineDispenser7.connect(p=1.)
network.add(dopamineDispenser1)
network.add(dopamineDispenser2)
network.add(dopamineDispenser3)
network.add(dopamineDispenser4)
network.add(dopamineDispenser5)
network.add(dopamineDispenser6)
network.add(dopamineDispenser7)













network.run(1*ms, report='text')

'''What does the network look like?'''
import matplotlib.pyplot as plt
import numpy.random as random
import matplotlib.patches as mpatches

plt.scatter([0,1], [10, 10], c = 'r', s = 100) # input1Generator
plt.scatter([9,10], [10, 10], c = 'r', s = 100) # input2Generator
plt.scatter([0,1], [8, 8], c = 'b', s = 100) # input1
plt.scatter([9,10], [8, 8], c = 'b', s = 100) # input2

kernelPosX = random.randn(numberNeuronsKernel) + 5
kernelPosY = random.randn(numberNeuronsKernel) + 4

plt.scatter(kernelPosX[:int(percentExc*numberNeuronsKernel)], kernelPosY[:int(percentExc*numberNeuronsKernel)], c = 'g', s = 10) #kernelExc
plt.scatter(kernelPosX[int(percentExc*numberNeuronsKernel):], kernelPosY[int(percentExc*numberNeuronsKernel):], c = 'purple', s = 10) #kernelInh


plt.scatter([4.5,5.5], [0,0], c = 'orange', s = 100) # input1

red_patch = mpatches.Patch(color='r', label='Input generators')
blue_patch = mpatches.Patch(color='b', label='Inputs')
green_patch = mpatches.Patch(color='g', label='Kernel Exc')
purple_patch = mpatches.Patch(color='purple', label='Kernel Inh')
orange_patch = mpatches.Patch(color='orange', label='Output')
plt.legend(handles=[red_patch,blue_patch,green_patch,purple_patch,orange_patch])


plt.show()
