# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:43:00 2019

@author: 85010
"""
import re,os,sys,csv
from collections import Counter
pPath = re.sub(r'AAC$', '', os.path.split(os.path.realpath(__file__))[0])
sys.path.append(pPath)
import readFasta
import pandas as pd

def AAC(fastas, **kw):
	AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	#AA = 'ARNDCQEGHILKMFPSTWYV'
	encodings = []
	header = ['#']
	for i in AA:
		header.append(i)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		count = Counter(sequence)
		for key in count:
			count[key] = count[key]/len(sequence)
		code = [name]
		for aa in AA:
			code.append(count[aa])
		encodings.append(code)
	return encodings

fastas = readFasta.readFasta("train_P1.txt")
#kw=  {'path': "E:\S-sulfenylation11\2018_11_21最新数据\3_特征提取\AAC\\codes",'train':"data_train.txt",'label':"label_train.txt",'order':'ACDEFGHIKLMNPQRSTVWY'}
#kw=  {'path': "D:\\xw\\特征提取\\AAC\\codes",'train':"E:\\S-sulfenylation11\\AAC\\codes\\data_train_p.txt",'label':"E:\\S-sulfenylation11\\AAC\\codes\\label_test.txt",'order':'ACDEFGHIKLMNPQRSTVWY'}
kw=  {'path': r"AAC",'train':r"train_P1.txt",'order':'ARNDCQEGHILKMFPSTWYVX'}
data_AAC=AAC(fastas, **kw)
#AAC=data_AAC.to_list
AAC=pd.DataFrame(data=data_AAC)
AAC.to_csv('AAC_train_P1.csv')