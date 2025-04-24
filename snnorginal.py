import numpy as np
import time

# dead pixels
# (62,125)
# (63,125)

# Model1 hyperparameters
# layer 1 weights ~ N(.5,.1)
# layer 1 thresholds ~ N(.5,.2)
# layer 1 epochs 20
# layer 1 learning rate .001
# layer 1 threshold learning rate .5

# Model2 hyperparameters
# layer 1 weights ~ N(.5,.1)
# layer 1 thresholds ~ N(.5,.2)
# layer 1 epochs 20
# layer 1 learning rate .1
# layer 1 threshold learning rate .5

class SNN:

	def __init__(self, layer_1_thresholds, layer_2_thresholds, layer_1_weights, layer_2_weights, accumulator_size=10):
		if isinstance(layer_1_weights, str):
			self._layer_1_weights = np.load(layer_1_weights)
		else:
			self._layer_1_weights = layer_1_weights
			self._layer_1_weights[self._layer_1_weights<=0] = 1e-7
			self._layer_1_weights /= np.linalg.norm(self._layer_1_weights, axis=0)

		if isinstance(layer_2_weights, str):
			self._layer_2_weights = np.load(layer_2_weights)
		else:
			self._layer_2_weights = layer_2_weights
			self._layer_2_weights[self._layer_2_weights<=0] = 1e-7
			self._layer_2_weights /= np.linalg.norm(self._layer_2_weights, axis=0)

		if isinstance(layer_1_thresholds, str):
			self._layer_1_thresholds = np.load(layer_1_thresholds)
		else:
			self._layer_1_thresholds = layer_1_thresholds

		if isinstance(layer_2_thresholds, str):
			self._layer_2_thresholds = np.load(layer_2_thresholds)
		else:
			self._layer_2_thresholds = layer_2_thresholds

		self._infer_layer_1_spikes = np.zeros((accumulator_size, self._layer_1_thresholds.shape[0]), dtype=np.int32)
		self._infer_current_frame = 0
		self._accumulator_size = accumulator_size


	def get_weights(self, layer=0):
		if layer == 0:
			return self._layer_1_weights
		else:
			return self._layer_2_weights


	def infer(self, input_frame):
		'''
		Computes the output spikes of layer 1 and 2 for a given sample.
		'''

		# Compute the membrane potential of neurons of layer 1
		layer_1_potentials = np.dot(input_frame.reshape((180*240,)), self._layer_1_weights)

		# Compute output spikes of layer 1
		layer_1_activations = layer_1_potentials>self._layer_1_thresholds
		self._infer_layer_1_spikes[self._infer_current_frame,:] = layer_1_activations

		# Compute the membrane potential of neurons of layer 2
		layer_2_potentials = np.dot(np.sum(self._infer_layer_1_spikes, axis=0), self._layer_2_weights)
		# Compute output spikes of layer 1
		layer_2_activations = layer_2_potentials>self._layer_2_thresholds

		# Switch to next frame (for the next call to infer)
		self._infer_current_frame = (self._infer_current_frame + 1)%self._accumulator_size

		# Return output spikes of both layers
		return layer_1_activations, layer_2_activations


	def train_snn(self, input_data, layer_1_epochs, layer_1_learning_rate, layer_1_lr_th,
					layer_2_epochs, layer_2_learning_rate, layer_2_lr_th, history_size=10):
		'''
		Layer-wise training of the network for a given training set.
		'''
		# train first layer
		l1_updates = np.zeros(self._layer_1_thresholds.shape, dtype=np.int32)
		for i in range(layer_1_epochs):
			for sample in input_data:
				# Compute active neurons (i.e. neurons that spike)
				activations = np.dot(sample.reshape((sample.shape[0]*sample.shape[1],)), self._layer_1_weights)
				actives = np.where(activations > self._layer_1_thresholds)[0]
				if len(actives) > 0:
					# Winner-takes-all: active neuron with highest potential wins the update
					winner = actives[np.argmax(activations[actives])]
					l1_updates[winner] += 1
					# Hebbian learning rule
					x = self._layer_1_weights[:, winner].copy()
					self._layer_1_weights[:, winner] += layer_1_learning_rate * sample.reshape((sample.shape[0]*sample.shape[1],))
					print(self._layer_1_weights[:, winner][self._layer_1_weights[:, winner] != x].shape)
					print("----------------------------------------------------------")
					# Weight normalization
					self._layer_1_weights[:, winner] /= np.linalg.norm(self._layer_1_weights[:, winner])
					# Homeostasis: increase the threshold of the winner
					self._layer_1_thresholds[winner] = activations[winner]
# previous version of homeostasis
#					self._layer_1_thresholds[winner] *= 1. + layer_1_lr_th
#					w_th = self._layer_1_thresholds[winner]
#					self._layer_1_thresholds[...] *= 1 - layer_1_lr_th/(len(self._layer_1_thresholds)-1)
#					self._layer_1_thresholds[winner] = w_th
				else:
					# Homeostasis: decrease all thresholds if no neuron was active
					self._layer_1_thresholds[...] *= 1 - layer_1_lr_th/len(self._layer_1_thresholds)
			#print(self._layer_1_thresholds, activations)
		print(l1_updates)

		# set thresholds of layer 1 for inference
		self._layer_1_thresholds[...] = 0.
		for sample in input_data:
			activations = np.dot(sample.reshape((sample.shape[0]*sample.shape[1])), self._layer_1_weights)
			self._layer_1_thresholds[...] = np.maximum(self._layer_1_thresholds, activations)
		self._layer_1_thresholds *= .95

		# train second layer
		l2_updates = np.zeros(self._layer_2_thresholds.shape, dtype=np.int32)
		# history of activations
		layer_1_history = np.zeros((history_size, self._layer_1_thresholds.shape[0]), dtype=np.int32)
		current_history = 0

		for i in range(layer_2_epochs):
			for sample in input_data:
				# Inference of layer 1
				layer_1_spikes = np.dot(sample.reshape((sample.shape[0]*sample.shape[1],)), self._layer_1_weights) >= self._layer_1_thresholds
				layer_1_history[current_history] = layer_1_spikes
				current_history = (current_history+1)%history_size
				# Compute active neurons in layer 2
				activations = np.dot(np.sum(layer_1_history, axis=0), self._layer_2_weights)
				actives = np.where(activations > self._layer_2_thresholds)[0]
				#print(layer_1_spikes, actives)
				if len(actives) > 0:
					# Winner takes all
					winner = actives[np.argmax(activations[actives])]
					l2_updates[winner] += 1
					# Hebbian weight update
					self._layer_2_weights[layer_1_spikes, winner] += layer_2_learning_rate
# Previous version: negative update of non-winner seurons
#					self._layer_2_weights[np.logical_not(layer_1_spikes), winner] -= layer_2_learning_rate
					# Weight normalization
					self._layer_2_weights[:, winner] /= np.linalg.norm(self._layer_2_weights[:, winner])
					# Homeostasis: update thresholds
					self._layer_2_thresholds[winner] = activations[winner]
# Previous version of homeostasis
#					self._layer_2_thresholds[winner] *= 1. + layer_2_lr_th
#					w_th = self._layer_2_thresholds[winner]
#					self._layer_2_thresholds[...] *= 1 - layer_2_lr_th/(len(self._layer_2_thresholds)-1)
#					self._layer_2_thresholds[winner] = w_th
				else:
					self._layer_2_thresholds[...] *= 1 - layer_2_lr_th/len(self._layer_2_thresholds)

		# Set thresholds of layer 2 for inference
		layer_1_history[...] = 0
		self._layer_2_thresholds[...] = 0
		for sample in input_data:
			layer_1_spikes = np.dot(sample.reshape((sample.shape[0]*sample.shape[1])), self._layer_1_weights) >= self._layer_1_thresholds
			layer_1_history[current_history] = layer_1_spikes
			current_history = (current_history+1)%history_size
			activations = np.dot(np.sum(layer_1_history, axis=0), self._layer_2_weights)
			self._layer_2_thresholds[...] = np.maximum(self._layer_2_thresholds, activations)
		self._layer_2_thresholds *= .95
