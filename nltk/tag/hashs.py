# -*- coding: utf-8 -*-

from nltk import korChar
from nltk.tokenize import word_tokenize,syllable_tokenize
import re

def load(path, filename):
	"""
	"""
	#print("loading hash: %s %s" % (path, filename) )

	filename=path+filename
	f = open(filename, "rt")
	keys = f.read().splitlines()			# remove newline character 
	f.close()

	return keys


def hash_key(hashs, idx):
	if idx < 0 or idx >= len(hashs) :
		assert("hash index out og bounds")

	return hashs[idx]


def hash_index(hashs, key_list):
	key_indicies = list()
	for k in key_list:
		key_indicies.append(hashs.index(k))
	
	return key_indicies
