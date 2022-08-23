# -*- coding: utf-8 -*-

from nltk import korChar
from nltk.tokenize import word_tokenize,syllable_tokenize
import re
import numpy as np
import math
import copy


def nn_lookup(dest, dest_stride, word_weights, word_size, max_word_idx, word_indices, nbr_word, pad_idx, nbr_pad) :
	if pad_idx < 0 or pad_idx >= max_word_idx :
		assert("lookup: padding index out of range");
	
	for i in range(nbr_pad):
		dest[i] = copy.deepcopy(word_weights[pad_idx])

	for i in range(nbr_word) :
		word_idx = word_indices[i]
		if word_idx < 0 or word_idx >= max_word_idx : 
			assert("lokup: index out of rnage")
		dest[(i+nbr_pad)] = copy.deepcopy(word_weights[word_idx])

	for i in range(nbr_pad) : 
		dest[(i+nbr_pad+nbr_word)] = copy.deepcopy(word_weights[pad_idx])

def nn_linear1(output, output_size, weights, biases, inputs, idx, window_size):
	for i in range(output_size): 
		z = biases[i] if biases[i] else 0;
		w = weights[i]
		in_context = copy.deepcopy(inputs[idx])
		for j in range(1, window_size):
			in_context = np.concatenate([in_context, inputs[idx+j]])
		z += np.dot(in_context, w)	
		output[i] = z

def nn_linear2(output, idx, output_size, weights, biases, inputs, input_size):
	for i in range(output_size): 
		z = biases[i] if biases[i] else 0;
		z += np.dot(inputs, weights[i])	
		output[idx][i] = z


def nn_hardtanh(outputs, inputs, size):
	for idx in range(size):
		if inputs[idx] >= -1 and inputs[idx] <= 1 :
			outputs[idx] = inputs[idx]
		elif inputs[idx] < -1:
			outputs[idx] = -1
		else:
			outputs[idx] = 1


def nn_viterbi(path, init, transition, emission, N, T) :
	'''
	init: 1 * 17
	transition: 17 * 17
	emission: output states T*N
	N: 품사 수
	T: 문장 길이 
	'''
	deltap = np.zeros(N)
	delta = np.zeros(N)
	phi = np.empty((T,N))

	for i in range(N) :
		deltap[i] = init[i] + emission[0][i]

	for t in range(1, T) :
		deltan = delta
		for j in range(N) :
			max_value = -math.inf
			max_idx = 0
			for i in range(N) :
				z = deltap[i] + transition[j][i]
				if z > max_value :
					max_value = z
					max_idx = i

			delta[j] = max_value + emission[t][j]
			phi[t][j] = int(max_idx)

		delta = deltap
		deltap = deltan
	
	max_value = -math.inf
	max_idx = 0
	for j in range(N) :
		if deltap[j] > max_value :
			max_value = deltap[j]
			max_idx = j
	path[T-1] = int(max_idx)
	
	# back tracking
	for t in range(T-2, -1, -1):
		path[t] = int(phi[t+1][int(path[t+1])])

