# -*- coding: utf-8 -*-

from nltk import korChar
from nltk.tokenize import word_tokenize,syllable_tokenize
import re

#추론 결과파일 이름 중복되지 않게
# filename0.result
def return_no_duplicated_filename(filename):
	import os
	if not os.path.exists(filename):
		return filename
	else:
		count = 0
		origin_filename = filename
		while True:
			tmp_filename = origin_filename.split('.')
			filename = tmp_filename[0] + str(count) +'.'+ tmp_filename[1]
			if not os.path.exists(filename):
				return filename
			count += 1

#CRF_lib.py로
def return_emjeol_list_from_file(filename):
	with open(filename,'r') as f:
		sentences = f.readlines()

		sentence_lists = list()
		word_list = list()
		for sentence in sentences:
			word_list = word_tokenize(sentence,'korean')
			syllable_list = list()
			for word in word_list:
				syllable_list += syllable_tokenize(word, 'korean')
			sentence_lists.append(syllable_list)

	return sentence_lists
		

def make_emjeol_file(sentense_filename):
	'''
	우리는 하나다.
	---->
	우
	리
	는
	하
	나
	다
	'''
	f_length = 0
	file_name = sentense_filename.split('.')[0]+'.gld'
	#with open(sentense_filename,'r',encoding='cp949') as fr:
	with open(sentense_filename,'r') as fr:
		with open(file_name,'w') as fw:
			line = fr.readline()
			result = list()
			word_list = list()
			while(line):
				word_list = word_tokenize(line,'korean')
				for word in word_list:
					syllable_list = syllable_tokenize(word,'korean')
					for emjeol in syllable_list:
						fw.write(emjeol+'\n')
						f_length+=1
				fw.write('\n')
				f_length+=1
				line = fr.readline()
	return file_name, f_length 		
	


def return_converted_word_from_emjeol(Y):
	word_Y = list()
	word_str = ""
	morph_str = ""	

	for y in Y:
		element = y.split('\t')
		is_first = element[0]
		if is_first == '1':
			word_Y.append(word_str + '\t' + morph_str.rstrip('+'))
			word_str = ""
			morph_str = ""
		word_str += element[1]
		morph_str += element[1]+'/'+element[2] +'+'
	word_Y.append(word_str + '\t' + morph_str.rstrip('+'))
	word_Y.pop(0)	
	return word_Y


# CRF_lib로 빼자
def write_inference_file_result(output_filename, Y_lists, wdsep):

	with open(output_filename,'w') as f:
		for Y_list in Y_lists:	
			print(Y_list)
			Sen_list = make_word_pos_list(Y_list, wdsep)
			for (e, t) in Sen_list:
				f.write(e + '\t' + t + '\n')
			f.write('\n')

	print('result prediction file:',output_filename)


	
def make_word_pos_list(Y_list, wdsep):
	'''
	음절로 된 태깅 결과를 형태소/어절 단위 태깅 결과로 변환한다.
	형식은 다음과 같다
	(morph, pos, 0/1)
	여기서 어절의 시작은 1, 아니면 0이다.

	:: TODO : 
	영어가 붙어 나오는 경우에 대해서 대처해야 한다.
	'''
	morph = ''
	pos = ''
	word = list()
	senL = list()
	sym_begin = False

	#morph = Y_list[0][0]
	#pos = Y_list[0][1]
	morph = '  '
	pos = '  '
	for i in range(0, len(Y_list)):
		print(morph, ' ', pos, '::', Y_list[i][0], ' ', Y_list[i][1])
		# 이전 음절에 연결하여 형태소 생성
		if pos[1] == 'S' and Y_list[i][1][1] != 'S' and pos[0] == Y_list[i][1][0]:
			morph = morph + Y_list[i][0]
			#pos = Y_list[i-1][1]
		elif Y_list[i][1] != 'CO' and pos == Y_list[i][1]:
			morph = morph + Y_list[i][0]
			#pos = Y_list[i-1][1]
		elif Y_list[i][1] == 'CO' :
			ch = korChar.kor_split(Y_list[i][0])
			print('\t', ch[0], ' ', ch[1], ' ', ch[2])
			if ch[2] in ['ㄴ','ㄹ','ㅁ']:
				m = korChar.kor_join(ch[0], ch[1], '')
				if m in ['하', '되']:
					senL.append((morph, pos))
					morph = m
					pos = 'XV' if pos[0] == 'N' else 'VB'
				elif m in ['이']:
					senL.append((morph, pos))
					morph = m
					pos = 'VB'
				elif pos in ['NS', 'NN', 'VB']:
					morph = morph + m
				else:
					# 불규칙 용언에 대한 처리를 해야 한다.
					senL.append((morph, pos))
					morph = m
					pos = 'VB'
				senL.append((morph, pos))
				morph = ch[2]
				pos = 'EE'
			elif ch[2] == 'ㅆ':
				if ch[1] == 'ㅕ':
					m = korChar.kor_join('ㅇ', 'ㅓ', ch[2])
					morph = morph +korChar.kor_join(ch[0], 'ㅣ', '') 
				elif ch[1] == 'ㅝ':
					m = korChar.kor_join('ㅇ', 'ㅓ', ch[2])
					morph = morph +korChar.kor_join(ch[0], 'ㅜ', '') 
				elif ch[1] == 'ㅙ':
					m = korChar.kor_join('ㅇ', 'ㅓ', ch[2])
					senL.append((morph, pos))
					morph = korChar.kor_join(ch[0], 'ㅚ', '') 
					pos = 'XV' if pos[0] == 'N' else 'VB'
				elif ch[1] == 'ㅐ':
					m = korChar.kor_join('ㅇ', 'ㅏ', ch[2])
					senL.append((morph, pos))
					morph = korChar.kor_join(ch[0], 'ㅏ', '') 
					pos = 'XV' if pos[0] == 'N' else 'VB'
				else:
					m = korChar.kor_join('ㅇ', ch[1], ch[2])
					morph = morph + ch[0]
				#pos = Y_list[i-1][1]
				senL.append((morph, pos))
				morph = m
				pos = 'EE'
			elif ch[2] == '':
				if ch[1] =='ㅕ':	# 려
					morph = morph +korChar.kor_join(ch[0], 'ㅣ', '') 
					m = korChar.kor_join('ㅇ', 'ㅓ', '')
					senL.append((morph, pos))
					senL.append((m, 'EE'))
					morph = ''
					pos = 'VB'
				elif ch[1] =='ㅐ':	# 해
					senL.append((morph, pos))
					morph = korChar.kor_join(ch[0], 'ㅏ', '') 
					m = korChar.kor_join('ㅇ', 'ㅏ', '')
					senL.append((morph, pos))
					senL.append((m, 'EE'))
					morph = ''
					pos = 'EE'
				elif ch[1] == 'ㅝ':
					m = korChar.kor_join('ㅇ', 'ㅓ', ch[2])
					morph = morph +korChar.kor_join(ch[0], 'ㅜ', '') 
					senL.append((morph, pos))
					senL.append((m, 'EE'))
					morph = ''
					pos = 'EE'
			else:	
				senL.append((morph, pos))
				morph = Y_list[i][0]
				pos = Y_list[i][1]
		else:
			senL.append((morph, pos))
			morph = Y_list[i][0]
			pos = Y_list[i][1]
	senL.append((morph, pos))
	
	return senL
	


def write_inference_sentence_result(Y_list):
	'''
	wdsep가 True이면 NS, MAS를 이용하여 띄어쓰기를 한다.
	그렇지 않으면  JJ, EE만을 구별한다.
	'''
	for (e, t) in Y_list:	# 음절, 품사 	
		print(e, '\t', t)

	



def evaluation(prediction_filename,anwser_filename):
	
	
	
	#with open(anwser_filename,'r',encoding='cp949') as f:
	with open(anwser_filename,'r') as f:
		anwser_list = f.readlines()
		re_anwser = list() 
		for line in anwser_list:
			if line != '\n' and len(line.split('\t')) > 1:
				re_anwser.append(line)

	with open(prediction_filename,'r') as f2:
		pred_list = f2.readlines()
		re_pred_list = list()
		for line in pred_list:
			if line != '\n' and len(line.split('\t')) > 1:
				re_pred_list.append(line)
	from nltk.metrics import accuracy
#	print("P:",re_pred_list[i]+"A:",re_anwser[i])
	print('result:',accuracy(re_anwser,re_pred_list))



#CRF lib로
def is_meta_syllable(value):
	try:
		if len(value) == 0 or value == '' or value == ' ':
			return '휅'
		if korChar.num_syllable(value):
			return '1'
		if korChar.eng_syllable(value):
			return 'A'
		if korChar.hanja_syllable(value):
			return '家'
		return value
	except Exception as e:
		return '휅'


def return_rowNcol(element):
	return re.findall(r'-?\d+',element)


# 음절	형태소 형식의 파일에서
# 음절부분만 가져와 문장으로 만듦
def emjeol_to_sentense(filename):
	with open(filename,'r') as f:
	
	#with open(filename,'r',encoding='cp949') as f:
		data =f.readlines()
		sentense_list = list()
		sentense = ""
		for line in data:
			splited_line = line.split('\t')
			x = splited_line[0]
			if len(splited_line) == 0 or len(splited_line) == 1:
				sentense_list.append(sentense)
				sentense = ""
			else:
				sentense += x
		sentense_list.append(sentense)

	with open(filename.split('.')[0]+"_sentense.dat",'w') as f:
		for i in sentense_list:
			f.write(i+'\n')




if  __name__ == "__main__":
	file_name = "test.gld"

	data = "./no_batch_model0.result"
	data2 = "./batch_model0.result"
	test(data,data2)
	#emjeol_to_sentense(file_name)



