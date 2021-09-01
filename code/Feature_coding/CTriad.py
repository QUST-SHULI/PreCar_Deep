import re
import readFasta
import numpy as np
import scipy.io as sio
import pandas as pd

def CalculateKSCTriad(sequence, gap, features, AADict):
	res = []
	for g in range(gap+1):
		myDict = {}
		for f in features:
			myDict[f] = 0

		for i in range(len(sequence)):
			if i+gap+1 < len(sequence) and i+2*gap+2<len(sequence):
				fea = AADict[sequence[i]] + '.' + AADict[sequence[i+gap+1]]+'.'+AADict[sequence[i+2*gap+2]]
				myDict[fea] = myDict[fea] + 1

		maxValue, minValue = max(myDict.values()), min(myDict.values())
		for f in features:
			res.append((myDict[f] - minValue) / maxValue)

	return res

def CTriad(fastas, gap = 0, **kw):
	AAGroup = {
		'g1': 'AGV',
		'g2': 'ILFP',
		'g3': 'YMTS',
		'g4': 'HNQW',
		'g5': 'RK',
		'g6': 'DE',
		'g7': 'C'
	}

	myGroups = sorted(AAGroup.keys())

	AADict = {}
	for g in myGroups:
		for aa in AAGroup[g]:
			AADict[aa] = g

	features = [f1 + '.'+ f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

	encodings = []
	header = ['#']
	for f in features:
		header.append(f)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		if len(sequence) < 3:
			print('Error: for "CTriad" encoding, the input fasta sequences should be greater than 3. \n\n')
			return 0
		code = code + CalculateKSCTriad(sequence, 0, features, AADict)
		encodings.append(code)

	return encodings
#fastas = readFasta.readFasta("C:\\Users\\Administrator\\Desktop\\Protein sequence\\Ion channel.txt")
#kw=  {'path': "E:\\examples\\",'train':"train-protein.txt",'label':"label.txt"}
#data_CTriad=CTriad(fastas, gap = 0,**kw)
#CTtriad1=np.array(data_CTriad)
#CTtriad2=CTtriad1[1:,1:]
#CTtriad3=CTtriad2.astype(np.float)
##CTtriad3.to_csv('data_CTriad_NR.csv')
#np.savetxt('data_CTriad_IC.csv',CTtriad3,delimiter=',')


kw=  {'path': r"train_P1.txt",}   
fastas1 = readFasta.readFasta(r"train_P1.txt")

result=CTriad(fastas1,gap = 1, **kw)
data1=np.matrix(result[1:])[:,1:]
data_=pd.DataFrame(data=data1)
data_.to_csv('CTriad_train_P1.csv')
