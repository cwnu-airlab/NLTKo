# -*- coding: utf-8 -*-


from nltk import korChar
from nltk.tokenize import word_tokenize,syllable_tokenize
import re
from nltk.tag import nn
import pickle
import numpy as np
from nltk.tag import hashs
import os
import nltk



class _get_postagger ():
	# size
	window_size = 0
	ll_word_vector_size = 50
	ll_word_max_idx = 0
	input_state_size = 0
	hidden_state_size = 0
	output_state_size = 0
	# weight
	ll_word_weight	= None
	l1_weight	= None
	l1_bias	= None
	l2_weight	= None
	l2_bias	= None
	viterbi_score_int	= None
	viterbi_score_trans = None

	#states
	input_states = None
	hidden_states = None
	output_states = None

	#
	labels = None
	token_idx = None
	mlabels = None
	morph_offset = None
	morph_nbr = None

	# verb dict
	verb_dict = dict()

	word_hash = None
	pos_hash = None

	# padding
	ll_word_padding_idx = 0			# word list에서 'PADDING'의 위치

	def __init__(self):
		'''

		'''
		path = os.path.dirname(nltk.tag.__file__)
		self.word_hash = hashs.load(path, "/hash/words.lst")
		self.ll_word_max_idx = len(self.word_hash)
		self.ll_word_padding_idx = self.word_hash.index("PADDING")

		# labels
		self.pos_hash = hashs.load(path, "/hash/pos.lst")

		# verb dict
		self.load_dict(path, "/hash/pos")

		# weights
		self.load(path, "/data/pos")

		
	def __del__(self):
		#print('Destructor called, Employee deleted.')
		a=None

	def load(self, path, subpath):
		# meta data
		with open(path+subpath+'-metadata.pickle', "rb") as f:
			meta = pickle.load(f)
			#print(meta)

		# weights
		network_info = np.load(path+subpath+'-network.npz')
		
		#print(network_info['hidden_weights'])	# l1. [[ 값들 ] ... [ 값들 ]]			300 * 250
		#print(network_info['output_weights'])	# l2. [[ 값들 ] ... [ 값들 ]]			17*300
		#print(network_info['hidden_bias'])	# l1 bias. [ 값들 ]
		#print(network_info['output_bias'])	# l2 bias. [ 값들 ]
		#print(network_info['word_window_size'])
		#print(network_info['input_size'])
		#print(network_info['hidden_size'])
		#print(network_info['output_size'])
		#print(network_info['padding_left'])						[24]
		#print(network_info['padding_right'])
		#print(network_info['transitions'])		# 0 번째가 viterbi score init, 그 다음이 1번부터 transitions. [[ 품사 수 ]]
		#print(network_info['feature_tables'])	# word_weight. [[[50 개 수],...[50개 수]]]
		
		self.window_size = int(network_info['word_window_size'])
		#self.ll_word_vector_size = 
		#self.ll_word_max_size = 
		self.input_state_size = int(network_info['input_size'])
		self.hidden_state_size = int(network_info['hidden_size'])
		self.output_state_size = int(network_info['output_size'])
		self.ll_word_weight = network_info['feature_tables'][0]
		self.l1_weight = network_info['hidden_weights']
		self.l1_bias = network_info['hidden_bias']
		self.l2_weight = network_info['output_weights']
		self.l2_bias = network_info['output_bias']
		self.viterbi_score_init = network_info['transitions'][0]
		self.viterbi_score_trans = network_info['transitions'][1:]

		self.input_states = np.zeros((self.input_state_size, self.ll_word_vector_size))
		self.hidden_states = np.zeros((self.hidden_state_size))

	def load_dict(self, path, subpath):
		"""
		"""
		subpath += "_vbs.lst"
		#print("loading verb dict: %s %s" % (path, subpath) )

		f = open(path+subpath, "rt")
		lines = f.read().splitlines()			# remove newline character 
		f.close()

		for line in lines :
			k, v = line.split('\t')
			if k in self.verb_dict:
				#print("load verb dict: key [%s] conflict!" % k)
				pass
			self.verb_dict[k] = v
		
	def final_touch(self, labels, tokens, token_indices):
		'''
		0:  CO
		1:  EE
		2:  IC
		3:  JJ
		4:  MA
		5:  MM
		6:  MS
		7:  NA
		8:  NN
		9:  NS
		10: SH
		11: SL
		12: SN
		13: SY
		14: VB
		15: XN
		16: XV

		어절 - 현태소, 동사 활용을 처리한다.
		이 땅에 태어났다.
		이	MM
		땅	NS
		에	JJ
		태	VB
		어	VB
		났	CO
		다	EE
		.	SY

		이	이/MM
		땅에	땅/NS+에/JJ
		태어났다.	태어나/VB+았/EE+다/EE+./SY

		TODO:
		1. 영어가 붙어 나오는 것에 대한 대처
		2. 수 처리
		3. 기호처리
		4. 원형복원 
		'''
		sent = list()
		morph = tokens[0]
		pos = labels[0]
		for idx in range(1, len(tokens)):
			#print(morph, ' ', pos, '::', tokens[idx], ' ', labels[idx] )
			ch = korChar.kor_split(tokens[idx])
			if ch[2] in ['ㅆ'] and labels[idx] in [1]:
				labels[idx] = 0

			if (pos in ['MS'] and labels[idx] in ['MA']) \
					or (pos in ['NS'] and labels[idx] in ['NN']):
				morph = morph + tokens[idx]
				pos = labels[idx]
			elif pos == labels[idx]:
				morph = morph + tokens[idx]
				pos = labels[idx]
			elif labels[idx] in ['CO'] or (tokens[idx] in ['운', '울', '웠'] and labels[idx] in ['EE']): #'CO'
				ch = korChar.kor_split(tokens[idx])
				#print('\t==>', ch[0], ch[1], ch[2])
				ch__ = korChar.kor_join(ch[0], ch[1], '')
				ch2 = ch[2]
				if ch2 in ['ㄴ', 'ㄹ', 'ㅁ', 'ㅆ']:
					if ch[1] in ['ㅓ', 'ㅜ', 'ㅝ', 'ㅞ'] and ch[2] in ['ㅆ']  : ch2 = '었'
					elif ch[1] in ['ㅏ', 'ㅗ', 'ㅘ'] and ch[2] in ['ㅆ']  : ch2 = '았'
					if ch__ in ['하', '되']:
						sent.append((morph, pos))
						morph = ch__
						pos = 'XV' if pos in ['NN'] else 'VB'
						sent.append((morph, pos))
						morph = ch2
						pos = 'EE'		# EE
					elif	ch__ in ['해', '돼']:					# 받침이 없는 경우 
						sent.append((morph, pos))
						ch__1 = 'ㅏ' if ch__ in ['해'] else 'ㅣ'
						morph = korChar.kor_join(ch[0], ch__1, '')
						pos = 'XV' if pos in ['NN'] else 'VB'
						sent.append((morph, pos))
						morph = '았'
						pos = 'EE'		# EE
					elif ch__ in ['이']:
						sent.append((morph, pos))
						morph = ch__
						pos = 'I'				# VB
						sent.append((morph, pos))
						morph = ch2
						pos = 'EE'		# EE
					elif pos in ['NS', 'NN']:
						morph = morph + ch__
						sent.append((morph, pos))
						morph = '이'
						pos = 'I'
						sent.append((morph, pos))
						morph = ch2
						pos = 'EE'		# EE
					elif pos in ['VB']:
						morph = morph + ch__
						sent.append((morph, pos))
						morph = ch2
						pos = 'EE'		# EE
					elif pos not in ['VB']: # 낼 사람
						sent.append((morph, pos))
						morph = ch__
						pos = 'VB'
						sent.append((morph, pos))
						morph = ch2
						pos = 'EE'		# EE
					elif pos not in ['NS','NN']: # 뭘?
						sent.append((morph, pos))
						morph = ch__
						pos = 'NS'
						sent.append((morph, pos))
						morph = ch2
						pos = 'JJ'		# JJ
				elif	ch__ in ['해', '돼']:					# 받침이 없는 경우 
					sent.append((morph, pos))
					morph = korChar.kor_join(ch[0], 'ㅏ', '')
					pos = 'XV' if pos in ['NN'] else 'VB'
					sent.append((morph, pos))
					morph = '어'
					pos = 'EE'		# EE
				elif	ch__ in ['혀']:					# 받침이 없는 경우 
					morph += ch__
					sent.append((morph, pos))
					morph = '어'
					pos = 'EE'		# EE
				elif	ch__ in ['라']:					# 받침이 없는 경우 
					morph += tokens[idx]
					sent.append((morph, pos))
					morph = '아'
					pos = 'EE'		# EE
				elif	ch__ in ['러']:					# 받침이 없는 경우 
					morph += tokens[idx]
					sent.append((morph, pos))
					morph = '어'
					pos = 'EE'		# EE
				else:
					morph = morph + tokens[idx]
					
			else:
				sent.append((morph, pos))
				morph = tokens[idx]
				pos = labels[idx]

		sent.append((morph, pos))

		# 동사 원형 검색
		Sent = list()
		for morph, pos in sent:
			if pos == 'VB' and morph in self.verb_dict:
				Sent.append((self.verb_dict[morph].split(' ')[0], pos))
			else:
				Sent.append((morph, pos))
		return Sent


	def forward(self, token_indices):
		nn.nn_lookup(self.input_states, self.ll_word_vector_size, self.ll_word_weight, self.ll_word_vector_size, self.ll_word_max_idx, token_indices, len(token_indices), self.ll_word_padding_idx, int((self.window_size-1)/2))

		self.output_states = np.zeros((len(token_indices), self.output_state_size))
		self.labels = np.zeros((len(token_indices)))

		for idx in range(len(token_indices)) :
			nn.nn_linear1(self.hidden_states, self.hidden_state_size, self.l1_weight, self.l1_bias, self.input_states, idx, self.window_size)
			nn.nn_hardtanh(self.hidden_states, self.hidden_states, self.hidden_state_size)
			nn.nn_linear2(self.output_states, idx, self.output_state_size, self.l2_weight, self.l2_bias, self.hidden_states, self.hidden_state_size)

		nn.nn_viterbi(self.labels, self.viterbi_score_init, self.viterbi_score_trans, self.output_states, self.output_state_size, len(token_indices))
	
		return list(self.labels)


	def pos_tag(self, tokens_list):
		token_indices = hashs.hash_index(self.word_hash, tokens_list)
		pos_labels = self.forward(token_indices)
		
		pos_labels_name = list()
		for i in pos_labels:
			 pos_labels_name.append(hashs.hash_key(self.pos_hash, int(i)))

		#for i in range(len(tokens_list)):
		#	 print(tokens_list[i], hashs.hash_key(self.pos_hash, int(pos_labels[i])))

		sent = self.final_touch(pos_labels_name, tokens_list, token_indices)

		return(sent)

	def pos_tag_with_verb_form(self, tokens_list):
		'''
		사랑/NN+하/XV ---> 사랑하/VB
		'''
		token_indices = hashs.hash_index(self.word_hash, tokens_list)
		pos_labels = self.forward(token_indices)
		
		pos_labels_name = list()
		for i in pos_labels:
			 pos_labels_name.append(hashs.hash_key(self.pos_hash, int(i)))

		#for i in range(len(tokens_list)):
		#	 print(tokens_list[i], hashs.hash_key(self.pos_hash, int(pos_labels[i])))

		tmp_sent = self.final_touch(pos_labels_name, tokens_list, token_indices)

		sent = list()
		for i in range(len(tmp_sent)):
			if tmp_sent[i][1] == 'XV':
				pass
			elif i<(len(tmp_sent)-1) and tmp_sent[i+1][1] == 'XV':
				sent.append((tmp_sent[i][0]+tmp_sent[i+1][0], 'VB'))
			else:
				sent.append((tmp_sent[i][0], tmp_sent[i][1]))

		return(sent)



	def word_segmentor(self, tokens_list):
		token_indices = hashs.hash_index(self.word_hash, tokens_list)
		pos_labels = self.forward(token_indices)

		sent = list()
		morph = ''
		sym_flag = 0
		for i in range(0, len(tokens_list)):			
			if pos_labels[i] in [13] and tokens_list[i] in ['(', '[','{','<']: # open symbols
				morph = tokens_list[i]
			elif int(pos_labels[i]) in [13] and tokens_list[i] in [')', '}', ']', '>']: # close symbols
				morph = morph + tokens_list[i]
				sent.append(morph)
				morph = ''
			elif pos_labels[i] in [13] and tokens_list[i] in ["'", '"', '`']: # open-close symbols
				if sym_flag == 1 :
					morph = morph + tokens_list[i]
					sym_flag = 0
				else :
					sym_flag = 1			# open
					morph = tokens_list[i]
			elif i > 0 and (pos_labels[(i-1)] == pos_labels[i] or \
					pos_labels[i] in [0,1,3,4,8,15,16] or\
					tokens_list[i] in ['이'] and pos_labels[i] in [14]):		# 조사, 어미, 접미사. XN, XV
				morph = morph + tokens_list[i]
			else:
				sent.append(morph)
				morph = tokens_list[i]


		return sent

