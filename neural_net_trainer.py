import pandas as pd
import numpy as np

class NeuralNetTrainer(object):
	LEARNING_RATE = 0.001
	MOMENTUM = 0.9
	MAX_SESSIONS = 1500
	DESIRED_ACCURACY = 90
	DESIRED_MSE = 0.001

	def __init__(self, NeuralNetwork):
		# Initialize all of our parameters
		self.NN = NeuralNetwork
		self.currentTrainingSession = 0
		self.learningRate = self.LEARNING_RATE
		self.momentum = self.MOMENTUM
		self.maxTraniingSessions = self.MAX_SESSIONS
		self.desiredAccuracy = self.DESIRED_ACCURACY
		self.useBatch = False
		self.traningSetAccuracy = 0
		self.validationSetAccuracy = 0
		self.generalizationSetAccuracy = 0
		self.trainingSetMSE = 0
		self.validationSetMSE = 0
		self.generalizationSetMSE = 0
		# Initialize the matrices for changes in weights
		self.dInputHidden = np.zeros(((self.NN).nInputNodes + 1, (self.NN).nHiddenNodes))
		self.dHiddenOutput = np.zeros(((self.NN).nHiddenNodes + 1, (self.NN).nOutputNodes))
		# Initialize vectors for error gradients
		self.hiddenErrorGradients = np.zeros((self.NN).nHiddenNodes + 1)
		self.outputErrorGradients = np.zeros((self.NN).nOutputNodes + 1)

	def setTrainingParameters(self, learningRate, momentum, useBatch):
		self.learningRate = learningRate
		self.momentum = momentum
		self.useBatch = useBatch
		return

	def setStoppingConditions(self, maxTrainingSessions, desiredAccuracy):
		self.maxTrainingSessions = maxTrainingSessions
		self.desiredAccuracy = desiredAccuracy
		return

	def _getOutputErrorGradient(self, trueValue, outputValue):
		return(outputValue * (1 - outputValue) * (desiredValue - outputValue))

	def _getHiddenErrorGradient(self, index):
		weightedSum = 0
		for i in range((self.NN).nOutputNodes):
			weightedSum += (self.NN).wHiddenOutput[index][i] * self.outputErrorGradients[i]
		return((self.NN).hiddenNeurons[index] * (1 - (self.NN).hiddenNeurons[index]) * weightedSum)

	def trainNetwork(self, trainingSet):
		# The trainingSet will consist of a set of dataEntryVectors

		return
	
	def runTrainingSession(self, dataEntryVector):
		# Each dataEntryVector will contain a set of inputs and a set of targets

		return

	def backpropagate(self, trueOutputs):
		# This function backpropagates the error to calculate the change in weights

		return

	def updateWeights(self):
		# This function uses the changes in weights to update them to their new values

		return







