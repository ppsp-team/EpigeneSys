Quick description :

GNW : Architecture for the Global Neuronal Workspace

STDP_Reward : Izhikevich's synpase model

XOR.ipynb : Sandbox for XOR function

Digits : Sandbox for MNIST digits recognition

ReinforceSynapse : Izhikevich's 2007 paper first experiment to test his model. In a large network of neurons, we artificially release dopamine when the synapse between neurons 0 and 1 is activated. The goal is to check that this synapse actually reinforces itself.

PavlovianConditioning : Izhikevich's 2007 paper second experiment to test his model. We have different interconnected groups of neurons, and we artificially realease dopaming when a neuron from group 1 is activated. The goal is to check that synapses coming from group 1 actually reinforce themselves more that the other ones.

SimpleConditioning : a slightly simpler version with only 2 neuron groups, cf SimpleConditioningDiagram.jpg


XOR.py : Implementation of the XOR function with respect to Izhikevich's 2007 model.
