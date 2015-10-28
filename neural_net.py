import pandas as pd
import numpy as np
from random import random

class NeuralNet(object):

	def __init__(self, nInputNodes, nHiddenNodes, nOutputNodes):
		# Take as inputs the count for input, hidden, and outer nodes
		self.nInputNodes = nInputNodes
		self.nHiddenNodes = nHiddenNodes
		self.nOutputNodes = nOutputNodes
		# Create nodes
		self.inputNeurons = np.zeros(self.nInputNodes + 1)
		self.hiddenNeurons = np.zeros(self.nHiddenNodes + 1)
		self.outputNerons = np.zeros(self.nOutputNodes)
		# Set the bias nodes
		self.inputNeurons[self.nInputNodes] = -1
		self.hiddenNeurons[self.nHiddenNodes] = -1
		# Create weights between each layer
		self.wInputHidden = np.zeros((self.nInputNodes + 1, self.nHiddenNodes))
		self.wHiddenOutput = np.zeros((self.nHiddenNodes + 1, self.nOutputNodes))

		self._initializeWeights()
		return
	
	# Private methods
	def _initializeWeights(self):
		# Fills wInputHidden and wHiddenOutput with random values
		rH = 1.0 / np.sqrt(self.nInputNodes)
		rO = 1.0 / np.sqrt(self.nHiddenNodes)

		for i in range(self.nInputNodes):
			for j in range(i, self.nHiddenNodes):
				self.wInputHidden[i][j] = ((random() * 100 + 1) * 2 * rH) - rH

		for i in range(self.nHiddenNodes):
			for j in range(i, self.nOutputNodes):
				self.wHiddenOutput[i][j] = ((random() * 100 + 1) * 2 * rO) - rO

		return

	def _activationFunction(self, x):
		# Sigmoid function
		return(1/(1 + np.exp(-x)))
	
	def _clampOutput(self, x):
		if x < 0.1:
			return 0
		elif x > 0.9:
			return 1
		else:
			return -1
	
	def _feedForward(self, inputVector):
		# Takes a set of inputs and feeds it through the network
		for i in range(self.nInputNodes):
			self.inputNeurons[i] = inputVector[i]

		for i in range(self.nHiddenNodes):
			self.hiddenNeurons[i] = 0
			for j in range(self.nInputNodes):
				self.hiddenNeurons[i] += self.inputNeurons[j] * self.wInputHidden[j][i]
			self.hiddenNeurons[i] = self._activationFunction(self.hiddenNeurons[i])

		for i in range(self.nOutputNodes):
			self.outputNeurons[i] = 0
			for j in range(self.nHiddenNodes):
				self.outputNeurons[i] += self.hiddenNeurons[j] * self.wHiddenOutput[j][i]
			self.outputNeurons[i] = self._activationFunction(self.outputNeurons[i])
		return

	# Other methods 
	def feedForward(self, inputVector):
		# Copies the output nodes so we can't directly influence them
		self._feedForward(inputVector)

		results = np.zeros(self.nOutputNodes)
		for i in range(self.nOutputNodes):
			results[i] = self._clampOutput(self.outputNeurons[i])

		return results

	# Metrics

	
	def getSetMSE(self, dataEntryVector):
		# Takes in a vector of the DataEntry objects and calculates the MSE
		# of the outputs relative to the targets
		mse = 0

		for i in range(len(dataEntryVector)):
			self._feedForward(dataEntryVector[i].inputs)
			for j in range(self.nOutputNodes):
				mse += np.pow((self.outputNeurons[j] - dataEntryVector[i].targets[j]), 2)

		return(mse / (self.nOutputNodes * len(dataEntryVector)))
	
	def getSetAccuracy(self, dataEntryVector):
		badResults = 0

		for i in range(len(dataEntryVector)):
			self._feedForward(dataEntryVector[i].inputs)
			goodResult = True
			for j in range(self.nOutputNodes):
				if self._clampOutput(self.outputNeurons[j]) != dataEntryVector[i].targets[j]:
					goodResult = False
			if not goodResult:
				badResults += 1

		return (100 - (badResults/len(dataEntryVector)) * 100)

