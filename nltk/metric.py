from nltk.metrics import confusionmatrix
import os
import argparse
from collections import defaultdict

def accuracy_score(true, pred):

	mat=confusionmatrix.ConfusionMatrix(true,pred)
	
	conf=mat._confusion
	total=0
	tp=0

	for r, tmp in enumerate(conf):
		for v, n in enumerate(tmp):
			if r==v:
				tp+=n
			total+=n

	return float(tp/total)

def recall_score(true,pred,avg='micro'):

	mat=confusionmatrix.ConfusionMatrix(true,pred)
	
	conf=mat._confusion
	indices=mat._indices
	values=mat._values
	total=0
		

	if len(values)==2:
		tp=0
		fn=0
		for r,i in enumerate(conf):
			for r2,v in enumerate(i):
				if r==0:
					continue
				elif r2==0:
					fn=v
				elif r==1:
					tp=v

		return float(tp/(tp+fn))


	c_tp=[]
	c_fn=[]
	recall_cls=[]

	for r, tmp in enumerate(conf):
		temp=0
		for v, n in enumerate(tmp):
			if r==v:
				c_tp.append(n)
			else:
				temp+=n
		c_fn.append(temp)

	if avg=='macro':

		for tmp in range(len(values)):
			try:
				recall_cls.append(float(c_tp[tmp]/(c_tp[tmp]+c_fn[tmp])))
			except:
				recall_cls.append(0)

		temp=0

		for tmp in recall_cls:
			temp+=tmp

		return float(temp/len(recall_cls))

	if avg=='micro':
		ja=0
		mo=0

		for tmp in range(len(values)):
			ja+=c_tp[tmp]
			mo+=c_tp[tmp]+c_fn[tmp]

		return float(ja/mo)

	else: 
		return "avg expect micro/macro"



def precision_score(true, pred,avg='micro'):


	mat=confusionmatrix.ConfusionMatrix(true,pred)
	
	conf=mat._confusion
	values=mat._values

	total=0

	if len(values)==2:
		tp=0
		fp=0
		for r,i in enumerate(conf):
			for r2,v in enumerate(i):
				if r2==0:
					continue
				elif r==0:
					fp=v
				elif r==1:
					tp=v

		return float(tp/(tp+fp))

	c_tp=list()
	c_fp=[0 for _ in range(len(values))]
	recall_cls=[]

	for r, tmp in enumerate(conf):
		for v, n in enumerate(tmp):
			if r==v:#tp
				c_tp.append(n)
			else:
				c_fp[v]+=n

	if avg=='macro':
		for tmp in range(len(values)):
			try:
				recall_cls.append(float(c_tp[tmp]/(c_tp[tmp]+c_fp[tmp])))
			except:
				recall_cls.append(0)
					
		temp=0

		for tmp in recall_cls:
			temp+=tmp

		return float(temp/len(recall_cls))


	elif avg=='micro':
		ja=0
		mo=0

		for tmp in range(len(values)):
			ja+=c_tp[tmp]
			mo+=c_tp[tmp]+c_fp[tmp]

		return float(ja/mo)

	else: 
		return "avg expect micro/macro"


def f1_score(true,pred,avg='micro'):

	if avg =='micro' or avg =='macro':

		precision=precision_score(true,pred,avg)
		recall=recall_score(true,pred,avg)
	else:
		return "avg expect micro/macro"

	return (((precision*recall)/(precision+recall))*2)




def pos_eval(fin):

	#temp=os.getcwd()+'/'+fin
	file=open(fin,'r').read()
	sents=file.split("\n\n")

	acc = defaultdict(float)
	t_avg = defaultdict(float)

	for sent in sents:
		lines=sent.split('\n')
		for line in lines:
			tot=line.split('\t')
			
			if line=='':continue

			wd=tot[0]
			gold=tot[1]
			pred=tot[2]
		
			acc['all']+=1
			gold_list=gold.split('+')
			pred_list=pred.split('+')

			t_avg["pr_all"]+=len(pred_list)
			t_avg["rc_all"]+=len(gold_list)

			if gold==pred:
				acc["true"]+=1
				t_avg['pr']+=len(pred_list)
				t_avg['rc']+=len(gold_list)
				continue
			else :
				intersect=0
				for g in gold_list:
					if not g in pred_list: continue
					intersect+=1
				t_avg['pr']+=intersect
				t_avg['rc']+=intersect
			

	t_avg['pr_result'] = t_avg['pr'] / t_avg['pr_all']
	t_avg['rc_result'] = t_avg['rc'] / t_avg['rc_all']

	return float(acc['true']/acc['all']) ,t_avg['pr_result'],t_avg['rc_result'], f1(t_avg['pr_result'], t_avg['rc_result'])


def f1(p, r):
	 return 2 * p * r / (p + r) if p + r else 0
		

