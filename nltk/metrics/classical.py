from nltk.metrics import confusionmatrix
import os
import argparse
from collections import defaultdict
from typing import List, Union
import torch
import numpy as np

class DefaultMetric:
	def __init__(self):
		pass

	def accuracy_score(self, true, pred):

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

	def recall_score(self, true, pred, avg='micro'):

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



	def precision_score(self, true, pred,avg='micro'):


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


	def f1_score(self, true, pred, avg='micro'):

		if avg =='micro' or avg =='macro':

			precision=self.precision_score(true,pred,avg)
			recall=self.recall_score(true,pred,avg)
		else:
			return "avg expect micro/macro"

		return (((precision*recall)/(precision+recall))*2)




	def pos_eval(self, fin):

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


	def f1(self, p, r):
		return 2 * p * r / (p + r) if p + r else 0
			

	def precision_at_k(self, true: List[int], pred: List[int], k: int) -> float:
		"""
		avg = ['micro', 'macro']
		"""
		
		relevant = 0

		if k > len(pred):
			raise ValueError("`k` is bigger than pred's length")

		pred = pred[:k]

		for t in true:
			if t in pred:
				relevant += 1
		
		
		return float(relevant/len(pred))

	def recall_at_k(self, true: List[int], pred: List[int], k: int) -> float:

		relevant = 0

		if k > len(pred):
			raise ValueError("`k` is bigger than pred's length")

		pred = pred[:k]

		for t in true:
			if t in pred:
				relevant += 1
		
		
		return float(relevant/len(true))

	def hit_rate_at_k(self, user: List[List[int]], pred: List[List[int]], k: int) -> float:
		hit = 0

		for u_list, p_list in zip(user, pred):
			try:
				p_list = p_list[:k]
			except:
				raise ValueError("`k` is bigger thant pred's length ")
			for u in u_list:
				if u in p_list:
					hit += 1
					break

		return float(hit/len(user))

	def mean_absolute_error(self, true: Union[torch.Tensor, np.ndarray], pred: Union[torch.Tensor, np.ndarray]) -> float:
		pass

	def root_mean_square_error(self, true: Union[torch.Tensor, np.ndarray], pred: Union[torch.Tensor, np.ndarray]) -> float:
		pass

def demo():
	y_pred = [5, 2, 4, 1, 3, 2, 5, 6, 7]
	y_true = [1, 3, 6, 7, 1, 5]

	user = [[5, 3, 2], [9, 1, 2], [3, 5, 6], [7, 2, 1]]
	h_pred = [[15, 6, 21, 3], [15, 77, 23, 14], [51, 23, 21, 2], [53, 2, 1, 5]]

	metric = DefaultMetric()
	print(metric.precision_at_k(y_true,  y_pred, 3))
	print(metric.recall_at_k(y_true,y_pred, 3))
	print(metric.hit_rate_at_k(user, h_pred, 2))

if __name__=="__main__":
	demo()