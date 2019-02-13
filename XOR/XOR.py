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
El = -60*mV
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

# Setting the stage
network = Network()

indices = np.array([0])
times = np.array([0])*ms



'''Simplest setup: bits encoded as bits, not inferred from the activity in an assembly of neurons'''
# Inputs: two neurons, one for 0, one for 1 
input1Generator = SpikeGeneratorGroup(2, indices, times)
Network.add(input1Generator)
input2Generator = SpikeGeneratorGroup(2, indices, times)
Network.add(input2Generator)

input1 = NeuronGroup(numberNeuronsKernel, 
                    ''' dv/dt=(-gl*(v-El)-(g_ampa*v+g_gaba*(v-er)))/memc : volt (unless refractory)
                        dg_ampa/dt = -g_ampa/tau_ampa : siemens
                        dg_gaba/dt = -g_gaba/tau_gaba : siemens''',
                    threshold='v>vt', 
                    reset='v = vr',
                    method='linear')
Network.add(input1)

input2 = NeuronGroup(numberNeuronsKernel, 
                    ''' dv/dt=(-gl*(v-El)-(g_ampa*v+g_gaba*(v-er)))/memc : volt (unless refractory)
                        dg_ampa/dt = -g_ampa/tau_ampa : siemens
                        dg_gaba/dt = -g_gaba/tau_gaba : siemens''',
                    threshold='v>vt', 
                    reset='v = vr',
                    method='linear')
Network.add(input2)




# Kernel: for now, all excitatory, do distinction between cortical layers
kernel = NeuronGroup(numberNeuronsKernel, 
                    ''' dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er)))/memc : volt (unless refractory)
                        dg_ampa/dt = -g_ampa/tau_ampa : siemens
                        dg_gaba/dt = -g_gaba/tau_gaba : siemens''',
                    threshold='v>vt', 
                    reset='v = vr',
                    method='linear')
Network.add(kernel)

percentInh = 0.2
percentExc = 1 - percentInh

# Ouput: for now, all excitatory, do distinction between cortical layers
output = NeuronGroup(2, 
                    ''' dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er)))/memc : volt (unless refractory)
                        dg_ampa/dt = -g_ampa/tau_ampa : siemens
                        dg_gaba/dt = -g_gaba/tau_gaba : siemens''',
                    threshold='v>vt', 
                    reset='v = vr',
                    method='linear')
Network.add(output)

# Synapse from the spike generator group to the actual input neurons
input1Synapse = Synapses(input1Generator, input1, on_pre = 'v += vt-El')
input2Synapse = Synapses(input2Generator, input1, on_pre = 'v += vt-El')
#input1Synapse.connect()
#input2Synapse.connect()
network.add(input1Synapse)
network.add(input2Synapse)

# Excitatory synapse in kernel connected to everyone 
exc2allDensity = 0.1
synapseExc = Synapses(kernel[:int(percentExc*numberNeuronsKernel)], kernel, on_pre =  'gAmpa += 0.3*nS')
#synapseExc.connect(p = exc2allDensity)
Network.add(synapseExc)

# Inhibitory synapses in kernel to inhibitory neurons
inh2inhDensity = 0.1
synapseInh2Inh = Synapses(kernel[int(percentExc*numberNeuronsKernel)+1:numberNeuronsKernel], kernel[int(percentExc*numberNeuronsKernel)+1:numberNeuronsKernel], on_pre =  'gGaba += 0.3*nS')
#synapseInh2Inh.connect(p = inh2inhDensity)
Network.add(synapseInh2Inh)

# Inhibitory synapses in kernel to excitatory neurons: this is where the plasticity happens
inh2excDensity = 0.1
synapseInh2Exc = Synapses(kernel[int(percentExc*numberNeuronsKernel)+1:numberNeuronsKernel], kernel[:int(percentExc*numberNeuronsKernel)],   
                                            '''
                                            s : 1
                                            dApre/dt=-Apre/tau_stdp : 1 (event-driven)
                                            dApost/dt=-Apost/tau_stdp : 1 (event-driven)
                                            ''', 
                                            on_pre='''  Apre += 1.
                                                        s = clip(s+(Apost-alpha_s)*eta, 0, gmax)
                                                        gGaba += s*nS''',
                                            on_post=''' Apost += 1.
                                                        s = clip(s+Apre*eta, 0, gmax)''')
#synapseInh2Exc.connect(p = inh2excDensity)
Network.add(synapseInh2Exc)


# Synapses between inputs and kernel: only connect to the excitatory neurons in the kernel
input1KernelSynapses = Synapses(input1, kernel[:int(percentExc*numberNeuronsKernel)],   '''
                                            s : 1
                                            dApre/dt=-Apre/tau_stdp : 1 (event-driven)
                                            dApost/dt=-Apost/tau_stdp : 1 (event-driven)
                                            ''', 
                                            on_pre='''  Apre += 1.
                                                        s = clip(s+(Apost-alpha_s)*eta, 0, gmax)
                                                        gAmpa += s*nS''',
                                            on_post=''' Apost += 1.
                                                        s = clip(s+Apre*eta, 0, gmax)''')

input2KernelSynapses = Synapses(input2, kernel[:int(percentExc*numberNeuronsKernel)],   '''
                                            s : 1
                                            dApre/dt=-Apre/tau_stdp : 1 (event-driven)
                                            dApost/dt=-Apost/tau_stdp : 1 (event-driven)
                                            ''', 
                                            on_pre='''  Apre += 1.
                                                        s = clip(s+(Apost-alpha_s)*eta, 0, gmax)
                                                        gAmpa += s*nS''',
                                            on_post=''' Apost += 1.
                                                        s = clip(s+Apre*eta, 0, gmax)''')
#input1KernelSynapses.connect()
#input2KernelSynapses.connect()
network.add(input1KernelSynapses)
network.add(input2KernelSynapses)

# Synapses between kernel and output

kernelExc2OutputSynapses = Synapses(kernel, output,   '''
                                            s : 1
                                            dApre/dt=-Apre/tau_stdp : 1 (event-driven)
                                            dApost/dt=-Apost/tau_stdp : 1 (event-driven)
                                            ''', 
                                            on_pre='''  Apre += 1.
                                                        s = clip(s+(Apost-alpha_s)*eta, 0, gmax)
                                                        gAmpa += s*nS''',
                                            on_post=''' Apost += 1.
                                                        s = clip(s+Apre*eta, 0, gmax)''')
kernelInh2OutputSynapses = Synapses(kernel, output,   '''
                                            s : 1
                                            dApre/dt=-Apre/tau_stdp : 1 (event-driven)
                                            dApost/dt=-Apost/tau_stdp : 1 (event-driven)
                                            ''', 
                                            on_pre='''  Apre += 1.
                                                        s = clip(s+(Apost-alpha_s)*eta, 0, gmax)
                                                        gGaba += s*nS''',
                                            on_post=''' Apost += 1.
                                                        s = clip(s+Apre*eta, 0, gmax)''')

#kernelExc2OutputSynapses.connect()
#kernelInh2OutputSynapses.connect()
network.add(kernelExc2OutputSynapses)
network.add(kernelInh2OutputSynapses)

'''What does the network look like?'''
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
