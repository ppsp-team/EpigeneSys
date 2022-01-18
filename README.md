# EpigeneSys

Goal of the project: modelling the Epigenesis of the Global Neuronal Workspace.

## Final paper [2018-2022]: Multi-Level Development of Cognitive Abilities in an Artificial Neural Network
* [```GNW```](https://github.com/ppsp-team/EpigeneSys/tree/master/GNW): Scripts for reproducing the results of the paper.

## Early experiments [2015-2018]:

### Implementation in [Brian 2](https://brian2.readthedocs.io/) of the Izhikevich's 2007 synapse model
* [```SynapseModel/STDP_Reward```](https://github.com/ppsp-team/EpigeneSys/blob/master/SynapseModel/STDP_Reward.ipynb): Brian 2 implementation of Izhikevich's synapse model as described in his 2007 paper (see equation below).

![equation](SynapseModel/equation.png)

Plots the evolution of the model parameters for a simple two-neuron network:
![stdp_model](SynapseModel/stdp_model.png)

### Synapse reinforcement
* [```ReinforceSynapse/ReinforceSynapse```](https://github.com/ppsp-team/EpigeneSys/blob/master/ReinforceSynapse/ReinforceSynapse.ipynb): Test of the first experiment in Izhikevich's 2007 paper. In a large network of neurons, release dopamine when the synapse between two given neurons is activated. The goal is to check whether this synapse is selectively reinforced.

![reinforce](ReinforceSynapse/reinforce.png)

### Conditioning
* [```Conditioning/SimpleConditioning```](https://github.com/ppsp-team/EpigeneSys/blob/master/Conditioning/SimpleConditioning.ipynb): Test of the second experiment in Izhikevich's 2007 paper. In a network composed of multiple small groups of neurons, release dopamine whenever neurons in the first group spike.
![conditioning](Conditioning/conditioning.png)

### Other early sandboxes
* [```MultiColumns/GNW```](https://github.com/ppsp-team/EpigeneSys/blob/master/MultiColumns/GNW.ipynb): Architecture for a structured multi-column network in Brian.
* [```XOR/XOR```](https://github.com/ppsp-team/EpigeneSys/tree/master/XOR): Sandbox for the XOR function.
* [```MultiColumns/Digits```](https://github.com/ppsp-team/EpigeneSys/blob/master/MultiColumns/Digits.ipynb): Sandbox for MNIST digits recognition.
