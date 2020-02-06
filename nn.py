from typing import Any, Mapping, List, Tuple, Optional, Union
import tensorflow as tf
import numpy as np
import random
import sys


# Base class for neural net layers
class Layer(object):
    def __init__(self) -> None:
        self.params: List[tf.Tensor]

    # Repeats a tensor as necessary and crops it to fit the specified output size
    @staticmethod
    def resize(tensor: tf.Tensor, newsize: int) -> tf.Tensor:
        if newsize < tensor.shape[1]:
            return tensor[ : , 0 : newsize]
        elif newsize > tensor.shape[1]:
            multiples = (newsize + int(tensor.shape[1]) - 1) // tensor.shape[1]
            tiled = tf.tile(tensor, [1, multiples])
            if newsize < tiled.shape[1]:
                return tiled[ : , 0 : newsize]
            else:
                return tiled
        else:
            return tensor

    # Gather all the variable values into an object for serialization
    def marshall(self) -> Mapping[str, Any]:
        return { 'params': [ p.numpy().tolist() for p in self.params ] }

    # Load the variables from a deserialized object
    def unmarshall(self, ob: Mapping[str, Any]) -> None:
        params = ob['params']
        if len(params) != len(self.params):
            raise ValueError('Mismatching number of params')
        for i in range(len(params)):
            self.params[i].assign(np.array(params[i]))

    # Returns the number of weights in this layer
    def weightCount(self) -> int:
        wc = 0
        for p in self.params:
            s = 1
            for d in p.shape:
                s *= d
            wc += s
        return wc


# Computes pair-wise products to reduce a vector size by 2
class LayerProductPooling(Layer):
    def __init__(self) -> None:
        self.params = []

    def act(self, x: tf.Tensor) -> tf.Tensor:
        half_size = int(x.shape[1]) // 2
        if int(x.shape[1]) != half_size * 2:
            raise ValueError("Expected an even number of input values")
        two_halves = tf.reshape(x, [-1, 2, half_size])
        return tf.multiply(two_halves[:, 0], two_halves[:, 1])


# Randomly connects inputs to outputs. (No weights)
class LayerShuffle(Layer):
	def __init__(self, size: int) -> None:
		self.indexes = [i for i in range(size)]
		random.seed(1234)
		random.shuffle(self.indexes)
		self.params = []

	def act(self, x: tf.Tensor) -> tf.Tensor:
		return tf.gather(x, self.indexes, axis = 1)


class LayerMaxPooling2d(Layer):
	def __init__(self) -> None:
		self.params = []

	def act(self, x: tf.Tensor) -> tf.Tensor:
		h = x.shape[1]
		w = x.shape[2]
		c = x.shape[3]
		cols = tf.reshape(x, (-1, h, int(w) // 2, 2, c))
		halfwidth = tf.math.maximum(cols[:,:,:,0,:], cols[:,:,:,1,:])
		rows = tf.reshape(halfwidth, (-1, int(h) // 2, 2, int(w) // 2, c))
		halfheight = tf.math.maximum(rows[:,:,0,:,:], rows[:,:,1,:,:])
		return tf.reshape(halfheight, (-1, int(h) // 2, int(w) // 2, c))


# A linear (a.k.a. "fully-connected", a.k.a. "dense") layer
class LayerLinear(Layer):
    def __init__(self, inputsize: int, outputsize: int) -> None:
        self.weights = tf.Variable(tf.random.normal([inputsize, outputsize], stddev = max(0.03, 1.0 / inputsize), dtype = tf.float64))
        self.bias = tf.Variable(tf.random.normal([outputsize], stddev = max(0.03, 1.0 / inputsize), dtype = tf.float64))
        self.params = [ self.weights, self.bias ]

    def act(self, x: tf.Tensor) -> tf.Tensor:
        return tf.add(tf.matmul(x, self.weights), self.bias)


# The input should be a single integer value
class LayerCatTable(Layer):
    def __init__(self, categories: int, outputsize: int) -> None:
        self.outputsize = outputsize
        self.weights = tf.Variable(tf.random.uniform([categories, outputsize], dtype = tf.float64))
        self.params = [ self.weights ]

    def act(self, x: tf.Tensor) -> tf.Tensor:
        return tf.reshape(tf.gather(self.weights, x), (-1, self.outputsize))


# Connects each output unit to a specified number of randomly-selected inputs * reps.
# Weights will be tied 'reps' times.
# Also adds the input to make it a residual layer
class LayerRandom(Layer):
	def __init__(self, inputsize: int, outputsize: int, connections: int, reps: int) -> None:
		if connections * reps + 1 > inputsize:
			raise ValueError("There are not enough inputs for so many connections")
		self.outputsize = outputsize
		self.reps = reps
		self.conns: List[List[int]] = []
		for i in range(outputsize):
			cands = [x for x in range(inputsize) if x != i]
			random.shuffle(cands)
			self.conns.append(cands[:connections * reps])
		self.weights = tf.Variable(tf.random_normal([outputsize, connections], stddev = 1.0 / (connections * reps), dtype = tf.float64))
		self.bias = tf.Variable(tf.random_normal([outputsize], stddev = 1.0 / (connections * reps), dtype = tf.float64))
		self.params = [ self.weights, self.bias ]

	def act(self, x: tf.Tensor) -> tf.Tensor:
		gathered_inputs = tf.gather(x, self.conns, axis = 1)
		tiled_weights = tf.tile(self.weights, [1, self.reps])
		prod = tf.multiply(gathered_inputs, tiled_weights)
		unbiased = tf.reduce_sum(prod, 2)
		return unbiased + self.bias + Layer.resize(x, self.outputsize)


# Connects each output unit to inputs with the offsets 1, 2, 4, 8, 16, ...
# Also adds the input to make it a residual layer
class LayerToroid_A(Layer):
	def __init__(self, inputsize: int, outputsize: int):
		if inputsize * 2 < outputsize or outputsize * 2 < inputsize:
			raise ValueError("input and output sizes cannot differ by more than a factor of 2")
		self.outputsize = outputsize
		connections_per_output = (inputsize - 1).bit_length()
		self.weights = tf.Variable(tf.random_normal([outputsize, connections_per_output], stddev = 1.0 / connections_per_output, dtype = tf.float64))
		self.bias = tf.Variable(tf.random_normal([outputsize], stddev = 1.0 / connections_per_output, dtype = tf.float64))
		self.params = [ self.weights, self.bias ]

		# Generate nodes for each output
		self.o_indexes: List[List[int]] = []
		for i in range(outputsize):
			i_indexes: List[int] = []
			j = 1
			while True:
				if j >= inputsize:
					break
				i_indexes.append((i + j) % inputsize)
				j *= 2
			self.o_indexes.append(i_indexes)

	def act(self, x: tf.Tensor) -> tf.Tensor:
		gathered_inputs = tf.gather(x, self.o_indexes, axis = 1)
		unbiased = tf.math.reduce_sum(gathered_inputs * self.weights, axis = 2)
		without_residual = unbiased + self.bias
		return without_residual + Layer.resize(x, self.outputsize)


# Connects each output unit to inputs with the offsets 1, 2, 4, 8, 16, ...
# Also adds the input to make it a residual layer
class LayerToroid_B(Layer):
	def __init__(self, inputsize: int, outputsize: int):
		if inputsize * 2 < outputsize or outputsize * 2 < inputsize:
			raise ValueError("input and output sizes cannot differ by more than a factor of 2")
		self.connections_per_output = (inputsize - 1).bit_length()
		self.outputsize = outputsize
		self.weights = tf.Variable(tf.random_normal([outputsize, 1], stddev = 1.0 / self.connections_per_output, dtype = tf.float64))
		self.bias = tf.Variable(tf.random_normal([outputsize], stddev = 1.0 / self.connections_per_output, dtype = tf.float64))
		self.params = [ self.weights, self.bias ]

		# Generate nodes for each output
		self.o_indexes: List[List[int]] = []
		for i in range(outputsize):
			i_indexes: List[int] = []
			j = 1
			while True:
				if j >= inputsize:
					break
				i_indexes.append((i + j) % inputsize)
				j *= 2
			self.o_indexes.append(i_indexes)

	def act(self, x: tf.Tensor) -> tf.Tensor:
		gathered_inputs = tf.gather(x, self.o_indexes, axis = 1)
		tiled_weights = tf.tile(self.weights, [1, self.connections_per_output])
		unbiased = tf.math.reduce_sum(gathered_inputs * tiled_weights, axis = 2)
		without_residual = unbiased + self.bias
		return without_residual + Layer.resize(x, self.outputsize)


# Connects each output unit to inputs with the offsets 1, 2, 4, 8, 16, ...
# Also adds the input to make it a residual layer
class LayerToroid_C(Layer):
	def __init__(self, inputsize: int, outputsize: int):
		if inputsize * 2 < outputsize or outputsize * 2 < inputsize:
			raise ValueError("input and output sizes cannot differ by more than a factor of 2")
		self.outputsize = outputsize
		connections_per_output = (inputsize - 1).bit_length()
		self.weights = tf.Variable(tf.random_normal([1, connections_per_output], stddev = 1.0 / connections_per_output, dtype = tf.float64))
		self.bias = tf.Variable(tf.random_normal([outputsize], stddev = 1.0 / connections_per_output, dtype = tf.float64))
		self.params = [ self.weights, self.bias ]

		# Generate nodes for each output
		self.o_indexes: List[List[int]] = []
		for i in range(outputsize):
			i_indexes = []
			j = 1
			while True:
				if j >= inputsize:
					break
				i_indexes.append((i + j) % inputsize)
				j *= 2
			self.o_indexes.append(i_indexes)

	def act(self, x: tf.Tensor) -> tf.Tensor:
		gathered_inputs = tf.gather(x, self.o_indexes, axis = 1)
		tiled_weights = tf.tile(self.weights, [self.outputsize, 1])
		unbiased = tf.math.reduce_sum(gathered_inputs * tiled_weights, axis = 2)
		without_residual = unbiased + self.bias
		return without_residual + Layer.resize(x, self.outputsize)


# A layer that connects each unit to log_2(n) other units, according to the edges in a hypercube
class LayerHypercube_A(Layer):
	# Let N = batch samples, F = feature maps, B = bits, I = inputs, O = outputs
	# incoming has shape=(N,I)
	def __init__(self, inputsize: int, outputsize: int, featuremaps: int):
		if inputsize > 2 * outputsize or outputsize > 2 * inputsize:
			raise ValueError("The inputs and outputs may not differ by more than a factor of 2")
		self.outputsize = outputsize
		self.featuremaps = featuremaps
		bits = (inputsize - 1).bit_length()
		self.weights = tf.Variable(tf.random_normal([featuremaps, bits, outputsize], stddev = 0.5 / bits, dtype = tf.float64)) # shape=(F,B,O)
		self.bias = tf.Variable(tf.random_normal([featuremaps, outputsize], stddev = 0.5 / bits, dtype = tf.float64)) # shape=(F,O)
		self.params = [ self.weights, self.bias ]

		# Regroup all the inputs to align with the edges in the hypercube
		self.fm: List[List[List[int]]] = [] # Construct shape=(F,B,O)
		for k in range(featuremaps):
			outer: List[List[int]] = []
			for j in range(bits):
				inner: List[int] = []
				for i in range(outputsize):
					inner.append((i ^ (1 << j)) % inputsize)
				outer.append(inner)
			self.fm.append(outer)

	def act(self, x: tf.Tensor) -> tf.Tensor:
		inputs_aligned_with_weights = tf.gather(x, self.fm, axis = 1) # shape=(N,F,B,O)

		# Multiply by the weights and add the bias
		prod = tf.multiply(inputs_aligned_with_weights, self.weights) # shape=(N,F,B,O)
		unbiased_net = tf.reduce_sum(prod, axis = 2) # shape=(N,F,O)
		biased_net = tf.add(unbiased_net, self.bias) # shape=(N,F,O)
		reshaped_biased_net = tf.reshape(biased_net, [-1, self.featuremaps * self.outputsize]) # shape=(N,F*O)

		# Make it residual by adding the input to the output
		tiled_incoming = tf.tile(Layer.resize(x, self.outputsize), [1, self.featuremaps]) # shape=(N,F*O)
		return tf.add(reshaped_biased_net, tiled_incoming) # shape=(N,F*O)


# A layer that connects each unit to log_2(n) other units, according to the edges in a hypercube
class LayerHypercube_B(Layer): # Shares weights over bits
	# Let N = batch samples, F = feature maps, B = bits, V = vertices
	# incoming has shape=(N,I)
	def __init__(self, inputsize: int, outputsize: int, featuremaps: int):
		if inputsize > 2 * outputsize or outputsize > 2 * inputsize:
			raise ValueError("The inputs and outputs may not differ by more than a factor of 2")
		self.outputsize = outputsize
		self.featuremaps = featuremaps
		self.bits = (inputsize - 1).bit_length()
		self.weights = tf.Variable(tf.random_normal([featuremaps, 1, outputsize], stddev = 0.5 / self.bits, dtype = tf.float64)) # shape=(F,1,O)
		self.bias = tf.Variable(tf.random_normal([featuremaps, outputsize], stddev = 0.5 / self.bits, dtype = tf.float64)) # shape=(F,O)
		self.params = [ self.weights, self.bias ]

		# Regroup all the inputs to align with the edges in the hypercube
		self.fm: List[List[List[int]]] = [] # Construct shape=(F,B,O)
		for k in range(featuremaps):
			outer: List[List[int]] = []
			for j in range(self.bits):
				inner: List[int] = []
				for i in range(outputsize):
					inner.append((i ^ (1 << j)) % inputsize)
				outer.append(inner)
			self.fm.append(outer)

	def act(self, x: tf.Tensor) -> tf.Tensor:
		inputs_aligned_with_weights = tf.gather(x, self.fm, axis = 1) # shape=(N,F,B,O)

		# Multiply by the weights and add the bias
		tiled_weights = tf.tile(self.weights, [1, self.bits, 1]) # shape=(F,B,O)
		prod = tf.multiply(inputs_aligned_with_weights, tiled_weights) # shape=(N,F,B,O)
		unbiased_net = tf.reduce_sum(prod, axis = 2) # shape=(N,F,O)
		biased_net = tf.add(unbiased_net, self.bias) # shape=(N,F,O)
		reshaped_biased_net = tf.reshape(biased_net, [-1, self.featuremaps * self.outputsize]) # shape=(N,F*O)

		# Make it residual by adding the input to the output
		tiled_incoming = tf.tile(Layer.resize(x, self.outputsize), [1, self.featuremaps]) # shape=(N,F*O)
		return tf.add(reshaped_biased_net, tiled_incoming) # shape=(N,F*O)


# A layer that connects each unit to log_2(n) other units, according to the edges in a hypercube
class LayerHypercube_C(Layer): # Shares weights over vertices
	# Let N = batch samples, F = feature maps, B = bits, V = vertices
	# incoming has shape=(N,I)
	def __init__(self, inputsize: int, outputsize: int, featuremaps: int):
		if inputsize > 2 * outputsize or outputsize > 2 * inputsize:
			raise ValueError("The inputs and outputs may not differ by more than a factor of 2")
		self.outputsize = outputsize
		self.featuremaps = featuremaps
		self.bits = (inputsize - 1).bit_length()
		self.weights = tf.Variable(tf.random_normal([featuremaps, self.bits, 1], stddev = 0.5 / self.bits, dtype = tf.float64)) # shape=(F,B,1)
		self.bias = tf.Variable(tf.random_normal([featuremaps, 1], stddev = 0.5 / self.bits, dtype = tf.float64)) # shape=(F,1)
		self.params = [ self.weights, self.bias ]

		# Regroup all the inputs to align with the edges in the hypercube
		self.fm: List[List[List[int]]] = [] # Construct shape=(F,B,O)
		for k in range(featuremaps):
			outer: List[List[int]] = []
			for j in range(self.bits):
				inner: List[int] = []
				for i in range(outputsize):
					inner.append((i ^ (1 << j)) % inputsize)
				outer.append(inner)
			self.fm.append(outer)

	def act(self, x: tf.Tensor) -> tf.Tensor:
		inputs_aligned_with_weights = tf.gather(x, self.fm, axis = 1) # shape=(N,F,B,O)

		# Multiply by the weights and add the bias
		tiled_weights = tf.tile(self.weights, [1, 1, self.outputsize]) # shape=(F,B,O)
		prod = tf.multiply(inputs_aligned_with_weights, tiled_weights) # shape=(N,F,B,O)
		unbiased_net = tf.reduce_sum(prod, axis = 2) # shape=(N,F,O)
		tiled_bias = tf.tile(self.bias, [1, self.outputsize]) # shape=(F,O)
		biased_net = tf.add(unbiased_net, tiled_bias) # shape=(N,F,O)
		reshaped_biased_net = tf.reshape(biased_net, [-1, self.featuremaps * self.outputsize]) # shape=(N,F*O)

		# Make it residual by adding the input to the output
		tiled_incoming = tf.tile(Layer.resize(x, self.outputsize), [1, self.featuremaps]) # shape=(N,F*O)
		return tf.add(reshaped_biased_net, tiled_incoming) # shape=(N,F*O)


class LayerConv(Layer):
    # filter_shape should take the form: (height, width, channels_incoming, channels_outgoing)
	def __init__(self, filter_shape: Tuple[int, ...]):
		spatial_size = 1
		for i in range(0, len(filter_shape) - 2):
			spatial_size *= filter_shape[i]
		self.weights = tf.Variable(tf.random_normal(filter_shape, stddev = 1.0 / spatial_size, dtype = tf.float64))
		self.params = [ self.weights ]

	def act(self, x: tf.Tensor) -> tf.Tensor:
		self.activation = tf.nn.convolution(x, self.weights, "SAME")
