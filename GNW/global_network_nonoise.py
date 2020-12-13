#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Original file: https://colab.research.google.com/drive/11UGbxieKt6ceb8QUb9Td2wTy3-Ux_G7L
"""

# Commented out IPython magic to ensure Python compatibility.
from skimage.util import img_as_float
from skimage.io import imread
import numpy as np
from skimage.transform import resize
from skimage.transform import rescale
from copy import copy
import pickle
import matplotlib
import os
from skimage.color import rgb2gray
from skimage.color import hsv2rgb
import random
import matplotlib.pyplot as plt
from brian2 import *
from IPython.display import clear_output
# %matplotlib inline
from joblib import Parallel, delayed

# Data preprocessing for S2 input
from random import shuffle
import pickle


# Declare the simulation code
def sim(rate_num, ratio, trace):
    print("Now processing: rate_num=", rate_num, "ratio=", ratio)
    count = np.empty((1000))
    with open('spikes_for_global_alt.pickle', 'rb') as f:
        t_neur, i_neur, labels = pickle.load(f)

    acc = 0
    for i in range(1000):
        this_sec = t_neur[t_neur > i][0]
        count[i] = i_neur[np.argwhere(t_neur == this_sec)[0][0]] * 1
        if labels[i] == count[i]:
          acc += 1 / 10
    # print('Accuracy of the local network: {:.3}%'.format(acc))

    labels1 = copy(labels)
    count1 = copy(count)
    d = np.arange(len(count1))
    shuffle(d)
    for i in range(len(count1)):
        labels[i] = labels1[d[i]]
        count[i] = count1[d[i]]

    i_neur = []
    t_neur = []

    dur = len(count)
    digits = 2
    num_nums_specific = dur // digits
    pic_dur = 1000  #total time slot for 1 picture (in ms)
    spikes_in_picture = 50

    # print('Numbers starting from:')
    # print(count[:10])
    # print('Labels starting from:')
    # print(labels[:10])

    per = np.empty((len(count)))

    z = 0
    for indx, cnt in enumerate(count):
        # num_fire = cnt // num_nums_specific
        num_fire = cnt

        times = np.zeros(spikes_in_picture)
        indices = np.zeros(spikes_in_picture)

        for cntcnt in range(spikes_in_picture):
            times[cntcnt] = 3 * cntcnt  # total spiking time is 50 * 3(2) ms
            indices[cntcnt] = num_fire

        i_neur = np.concatenate([i_neur, indices])
        t_neur = np.concatenate([t_neur, times + (z + 1) * pic_dur])  # first second - relaxation
        z += 1
        if z >= dur:
            break

    # print('Number of pictures: ', z)

    t_neur = t_neur * ms
    # print('Size of the input arrays: ', t_neur.shape, i_neur.shape)
    # print('Unique neuron indices firing: ', len(np.unique(i_neur)))

    # Trigger input
    spikes_in_picture_trig = 50
    i_trigger = np.zeros(dur*spikes_in_picture_trig)
    t_trigger = np.empty(dur*spikes_in_picture_trig)

    z = 0
    for cnt in range(dur):
        for cntcnt in range(spikes_in_picture_trig):
            t_trigger[z] = 100 + trace * pic_dur / 4. + (cnt + 1) * pic_dur + cntcnt
            z += 1

    t_trigger = t_trigger * ms

    # Parameters
    # Neurons
    gmax = .25
    taum = 10 * ms
    Ee = 0 * mV
    vt = -54 * mV
    vr = -60 * mV
    El = -74 * mV
    taue = 5 * ms

    Ei = -70 * mV

    # STDP
    taupre = 20 * ms
    taupost = taupre
    dApre = 1e-3  # initially 1e-3 change back!!!!!!!!! 0.01
    dApost = -dApre * taupre / taupost * 1.  # 1.05 initially what about 1.3(everything flips)?
    dApost *= gmax
    dApre *= gmax

    Egmax = 0.02  # 0.02
    Igmax = Egmax * 10 #10
    dApre_E = 1.5e-3 * Egmax  # 1e-3 (1.5e-3)
    dApre_I = 1.5e-3 * Igmax  # 1e-3 (1.5e-3)
    dApost_E = -dApre_E * taupre / taupost * 1.05
    dApost_I = -dApre_I * taupre / taupost * 1.05

    # Dopamine signaling
    tauc = 400 * ms  # initially 1000 ms
    taud = 200 * ms
    taus = 5 * ms  # initially 1ms
    epsilon_dopa = 5e-3

    # populations
    N = 100
    # N_E = int(N * 0.8)  # pyramidal neurons
    # N_I = int(N * 0.2)  # interneurons, 0.2
    N_E = int(N * ratio / 100)
    N_I = int(N * (100 - ratio) / 100)

    # external stimuli
    # rate = 10 * Hz # 10
    rate = rate_num * Hz
    C_ext = 800
    ge_ext = 2.2e-2

    # subpopulations
    p = 2
    f = 0.5
    N_sub = int(N_E * f)

    net = Network()

    eqs = '''
    dv/dt = (ge * (Ee-vr) + El - v) / taum : volt (unless refractory)
    dge/dt = -ge / taue : 1
    '''

    P_E = NeuronGroup(N_E, eqs, threshold='v>vt', reset='v = vr', refractory=1*ms, method='euler')
    P_E.v = vr
    net.add(P_E)
    if N_I > 0:
        P_I = NeuronGroup(N_I, eqs, threshold='v>vt', reset='v = vr', refractory=1 * ms, method='euler')
        P_I.v = vr
        net.add(P_I)

    # Input to memory

    input_spikes = SpikeGeneratorGroup(digits, i_neur, t_neur)
    net.add(input_spikes)

    input_mon = SpikeMonitor(input_spikes)
    net.add(input_mon)

    S_sp = Synapses(input_spikes, P_E, on_pre='v_post+=100*mV')
    S_sp.connect(condition='i == j // N_sub')
    net.add(S_sp)

    # Trigger input

    input_trigger = SpikeGeneratorGroup(1, i_trigger, t_trigger)
    net.add(input_trigger)

    trigger_mon = SpikeMonitor(input_trigger)
    net.add(trigger_mon)

    # Synapses within memory layer

    eqs_S_E = '''mode: 1
    dc/dt = -c / tauc : 1 (clock-driven)
    dd/dt = -d / taud : 1 (clock-driven)
    ds/dt = mode * c * d / taus : 1 (clock-driven)
    dApre/dt = -Apre / taupre : 1 (event-driven)
    dApost/dt = -Apost / taupost : 1 (event-driven)'''

    eqs_pre = '''
    ge += s
    Apre += dApre_E
    c = clip(c + mode * Apost, -Egmax, Egmax)
    s = clip(s + (1-mode) * Apost, 0, Egmax)
    '''

    eqs_post = '''Apost += dApost_E
    c = clip(c + mode * Apre, -Egmax, Egmax)
    s = clip(s + (1-mode) * Apre, 0, Egmax)
    '''

    eqs_S_I = '''mode: 1
    dc/dt = -c / tauc : 1 (clock-driven)
    dd/dt = -d / taud : 1 (clock-driven)
    ds/dt = mode * c * d / taus : 1 (clock-driven)
    dApre/dt = - (Apre + dApre_I/15) / taupre : 1 (event-driven)
    dApost/dt = - (Apost + dApre_I/15) / taupre : 1 (event-driven)'''

    # /15

    eqs_pre_I = '''
    ge -= s
    Apre += dApre_I
    c = clip(c + mode * Apost, -Igmax, Igmax)
    s = clip(s + (1-mode) * Apost, 0, Igmax)
    '''

    eqs_post_I = '''Apost += dApre_I
    c = clip(c + mode * Apre, -Igmax, Igmax)
    s = clip(s + (1-mode) * Apre, 0, Igmax)
    '''

    # E to E
    C_E_E = Synapses(P_E, P_E, model=eqs_S_E, on_pre=eqs_pre, on_post=eqs_post, method='euler')
    C_E_E.connect('i != j')
    C_E_E.mode = 0
    C_E_E.s = '0.01 * rand() * Egmax'
    C_E_E.c = 1e-10
    C_E_E.d = 0
    net.add(C_E_E)

    if N_I > 0:
        # E to I
        C_E_I = Synapses(P_E, P_I, model=eqs_S_E, on_pre=eqs_pre, on_post=eqs_post, method='euler')
        C_E_I.connect()
        C_E_I.mode = 0
        C_E_I.s = '0.01 * rand() * Egmax'
        C_E_I.c = 1e-10
        C_E_I.d = 0
        net.add(C_E_I)

        # I to I
        C_I_I = Synapses(P_I, P_I, model=eqs_S_I, on_pre=eqs_pre_I, on_post=eqs_post_I, method='euler')
        C_I_I.connect('i != j')
        C_I_I.mode = 0
        C_I_I.s = '0.01 * rand() * Igmax'
        C_I_I.c = 1e-10
        C_I_I.d = 0
        net.add(C_I_I)

        # I to E
        C_I_E = Synapses(P_I, P_E, model=eqs_S_I, on_pre=eqs_pre_I, on_post=eqs_post_I, method='euler')
        C_I_E.connect()
        C_I_E.mode = 0
        C_I_E.s = '0.01 * rand() * Igmax'
        C_I_E.c = 1e-10
        C_I_E.d = 0
        net.add(C_I_E)

    synapse_stdp_monitor = StateMonitor(C_E_E, ['s', 'c', 'd'], record=True, dt=100*ms)
    net.add(synapse_stdp_monitor)

    if N_I > 0:
        synapse_stdp_monitor1 = StateMonitor(C_I_E, ['s', 'c', 'd'], record=True, dt=100*ms)
        net.add(synapse_stdp_monitor1)

        synapse_stdp_monitor2 = StateMonitor(C_E_I, ['s', 'c', 'd'], record=True, dt=100*ms)
        net.add(synapse_stdp_monitor2)

        synapse_stdp_monitor3 = StateMonitor(C_I_I, ['s', 'c', 'd'], record=True, dt=100*ms)
        net.add(synapse_stdp_monitor3)

    # external noise
    C_P_E = PoissonInput(P_E, 'ge', C_ext, rate, 'ge_ext')
    net.add(C_P_E)
    if N_I > 0:
        C_P_I = PoissonInput(P_I, 'ge', C_ext, rate, 'ge_ext')
        net.add(C_P_I)

    # Motor cortex
    motor_output = NeuronGroup(1, model='''
                                        dv/dt = (ge * (Ee-vr) + El - v) / taum : volt (unless refractory)
                                        dge/dt = -ge / taue : 1
                                        dvtrig/dt = - vtrig / (50 * ms) : 1
                                        ''', threshold='v>vt', reset='v = vr', method='euler')
    motor_output.v = vr
    net.add(motor_output)

    mon = SpikeMonitor(motor_output)
    net.add(mon)

    S_out = Synapses(P_E, motor_output, model=eqs_S_E,
                       on_pre='''
                              Apre += dApre
                              c = clip(c + mode * Apost, -gmax, gmax)
                              s = clip(s + (1-mode) * Apost, 0, gmax)
                              ge += s * vtrig
                              ''',
                       on_post='''Apost += dApost
                              c = clip(c + mode * Apre, -gmax, gmax)
                              s = clip(s + (1-mode) * Apre, 0, gmax)
                              ''',
                       method='euler')

    S_out.connect()
    S_out.mode = 1
    # S_out.s = 1e-10
    S_out.c = 1e-10
    S_out.d = 0
    S_out.s = 0.5 * gmax  # 0.1
    net.add(S_out)

    synapse_stdp_monitor_out = StateMonitor(S_out, ['s', 'c', 'd'], record=True, dt=50*ms)
    net.add(synapse_stdp_monitor_out)

    check_trig = Synapses(input_trigger, motor_output, model='''''', on_pre='vtrig_post = 1', method='exact')
    check_trig.connect(p=1.)
    net.add(check_trig)

    # noise (motor)
    input_Poisson2 = PoissonGroup(1, rates=rate)
    net.add(input_Poisson2)

    S_poi2 = Synapses(input_Poisson2, motor_output, on_pre='v_post+=100*mV')
    S_poi2.connect(j='i')
    net.add(S_poi2)

    # Dopamine signaling section

    dopamine_times = TimedArray(np.r_[[0], labels], dt=pic_dur*ms) #to start at 1s, first second is for relexation

    dopamine = NeuronGroup(1, '''
                                dv/dt =  - v / (100 * ms) : volt (unless refractory)
                              ''',
                           threshold='v>100 * mV', refractory=pic_dur / 2 * ms,
                           reset='v=0 * mV', method='linear')
    dopamine.v = 0 * mV
    net.add(dopamine)
    dopamine_trigger = Synapses(motor_output, dopamine, model='''''', on_pre='v_post += 1.5*mV', method='exact') #change on lower value
    dopamine_trigger.connect(p=1.)
    net.add(dopamine_trigger)

    dopamine_monitor = SpikeMonitor(dopamine)
    net.add(dopamine_monitor)

    reward = Synapses(dopamine, S_out, model='''''',
                                 on_pre='''d_post += epsilon_dopa * (dopamine_times(t) * 2 - 1)''',
                                 method='exact')
    reward.connect()
    reward.delay='(rand() / 32.) * pic_dur *ms'
    net.add(reward)

    # monitors
    N_activity_plot = 20
    sp_E_sels = [SpikeMonitor(P_E[pi:pi + N_activity_plot]) for pi in range(0, p * N_sub, N_sub)]
    r_E_sels = [PopulationRateMonitor(P_E[pi:pi + N_sub]) for pi in range(0, p * N_sub, N_sub)]
    net.add(sp_E_sels)
    net.add(r_E_sels)
    if N_I > 0:
        sp_I = SpikeMonitor(P_I[:N_activity_plot])
        net.add(sp_I)

        r_I = PopulationRateMonitor(P_I)
        net.add(r_I)
    else:
        sp_I = 0
        r_I = 0

    # simulate, can be long >120s
    net.run(350 * second, report='stdout')

    # pickling

    dump = [r_I.t / second, r_I.smooth_rate(width=25 * ms) / Hz,
            r_E_sels[0].t / second, r_E_sels[0].smooth_rate(width=25 * ms) / Hz,
            r_E_sels[1].t / second, r_E_sels[1].smooth_rate(width=25 * ms) / Hz, sp_I.t / second,
            sp_I.i * 1, sp_E_sels[0].t / second, sp_E_sels[0].i * 1,
            sp_E_sels[1].t / second, sp_E_sels[1].i * 1,
            input_mon.t / second, input_mon.i * 1, trigger_mon.t / second, trigger_mon.i * 1,
            mon.t / second, mon.i * 1, dopamine_monitor.t / second, dopamine_monitor.i * 1, N_activity_plot,
            synapse_stdp_monitor_out.t/second, synapse_stdp_monitor_out.s.T/gmax]

    with open('data_rate{}_ratio{}_cond{}.pickle'.format(rate_num, ratio, trace), 'wb') as f:
        pickle.dump(dump, f)

if __name__ == '__main__':
    # Generate parameters for parallelization
    params = []
    #for rate_num in range(0, 11):
    rate_num = 0
    for ratio in range(10, 101, 10):
        for trace in [True, False]:  # False = Delay conditioning
            params.append((rate_num, ratio, trace))


    Parallel(n_jobs=-2, backend="multiprocessing")(delayed(sim)(rate_num, ratio, trace) for rate_num, ratio, trace in params)