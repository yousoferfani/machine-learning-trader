import pandas as pd
import numpy as np

class NeuralNet(object):
	def __init__(self, inner, hidden, outer):
		# Take as inputs the count for input, hidden, and outer nodes
		self.nInner = inner
		self.nHidden = hidden
		self.nOuter = outer
		# Create nodes
		self.inputNeurons = np.zeros(self.nInput + 1)
		self.hiddenNeurons = np.zeros(self.nHidden + 1)
		self.outputNerons = np.zeros(self.nOuter)
		# Set the bias nodes
		self.inputNeurons[self.nInput] = -1
		self.hiddenNeurons[self.nHidden] = -1
		# Create weights between each layer
		self.wInputHidden = np.zeros((self.nInput + 1, nHidden))
		self.wHiddenOutput = np.zeros((self.nHidden + 1, nOutput))

		self._initializeWeights()

	def _initializeWeights():
		# Fills wInputHidden and wHiddenOutput with random values



	def _activationFunction(self, x):
		# Sigmoid function
		return(1/(1 + np.exp(-x)))
