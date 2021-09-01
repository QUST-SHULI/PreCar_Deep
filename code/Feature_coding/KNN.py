import sys, os, re
import math
import numpy as np
import pandas  as pd
pPath = re.sub(r'codes$', '', os.path.split(os.path.realpath(__file__))[0])
sys.path.append(pPath)
import readFasta

def Sim(a, b):
	blosum62 = [
		[ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, 0],  # A
		[-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, 0],  # R
		[-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3, 0],  # N
		[-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3, 0],  # D
		[ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],  # C
		[-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, 0],  # Q
		[-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2, 0],  # E
		[ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, 0],  # G
		[-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3, 0],  # H
		[-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, 0],  # I
		[-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, 0],  # L
		[-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2, 0],  # K
		[-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, 0],  # M
		[-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, 0],  # F
		[-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, 0],  # P
		[ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2, 0],  # S
		[ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, 0],  # T 
		[-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, 0],  # W
       [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, 0],  # Y
		[ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, 0],  # V
		[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0],  # -
	]
	AA = 'ARNDCQEGHILKMFPSTWYV-'
	myDict = {}
	for i in range(len(AA)):
		myDict[AA[i]] = i

	maxValue, minValue = 11, -4
	return (blosum62[myDict[a]][myDict[b]] - minValue) / (maxValue - minValue)


def CalculateDistance(sequence1, sequence2):
	if len(sequence1) != len(sequence2):
		print(sequence1)
		print(sequence2)
		print('Error: inconsistent peptide length')
		sys.exit(1)
	distance = 1 - sum([Sim(sequence1[i], sequence2[i]) for i in range(len(sequence1))]) / len(sequence1)
	return distance


def CalculateContent(myDistance, j, myLabelSets):
	content = []
	myDict = {}
	for i in myLabelSets:
		myDict[i] = 0
	for i in range(j):
		myDict[myDistance[i][0]] = myDict[myDistance[i][0]] + 1
	for i in myLabelSets:
		content.append(myDict[myLabelSets[i]] / j)
	return content

def KNNpeptide(fastas, **kw):
	trainFile = kw['train']
	labelFile = kw['label']
	if trainFile == None or labelFile == None:
		print('Error: please specify the directory of train file ["--train"] and the label file ["--label"]')
		sys.exit(1)

	if os.path.exists(labelFile) == False:
		print('Error: the label file does not exist.')
		sys.exit(1)

	trainData = readFasta.readFasta(trainFile)
	with open(labelFile) as f:
		records = f.readlines()
	myLabel = {}
	for i in records:
		array = i.rstrip().split() if i.strip() != '' else None
		myLabel[array[0]] = int(array[1])
	myLabelSets = list(set(myLabel.values()))

	if len(trainData) != len(myLabel):
		print('ERROR: inconsistent sample number between train and label file.')
		sys.exit(1)

#	kValues = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15,
#			   0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]
	kNum = [2,4,8,16,32]
##    kNum = []
   # kNum=[2,4,8,16,32,64,128]
#	for i in kValues:
#		kNum.append(math.ceil(len(trainData) * i))

	encodings = []
	header = ['#']
	for k in kNum:
		for l in myLabelSets:
			header.append('Top' + str(k) + '.label' + str(l))
	encodings.append(header)
#    for k in kNum:
#	for l in myLabelSets:
#		header.append('Top' + str(k) + '.label' + str(l))
#   encodings.append(header)

	for i in fastas:
		name, sequence = i[0], i[1]
		code = [name]
		myDistance = []
		for j in range(len(trainData)):
			if name != trainData[j][0]:
				myDistance.append([myLabel[trainData[j][0]], CalculateDistance(trainData[j][1], sequence)])

		myDistance = np.array(myDistance)
		myDistance = myDistance[np.lexsort(myDistance.T)]

		for j in kNum:
			code = code + CalculateContent(myDistance, j, myLabelSets)
		encodings.append(code)

	return encodings
#fastas = readFasta.readFasta("data_test.txt")
#kw=  {'path': "D:\\xw\\3_特征提取\\KNN\\codes\\codes",'train':"data_test.txt",'label':"label_test.txt",'order':'ACDEFGHIKLMNPQRSTVWY'}
#kw=  {'path': "D:\\xw\\S-sulfenylation_特征提取\\KNN\\KNN0710\\codes",'train':"data_train.txt",'label':"label_train.txt",'order':'ACDEFGHIKLMNPQRSTVWY'}
fastas = readFasta.readFasta(r"train_P1.txt")
kw=  {'path': r"D:\代码\codes\KNN",'train':r"train_P1.txt",'label':r"label_train_P1.txt",'order':'ACDEFGHIKLMNPQRSTVWYX'}

data_KNN=KNNpeptide(fastas, **kw)

data_raw=data_KNN[1:]
data_new=np.matrix(data_raw)
data_knn=data_new[:,1:]
data_knn_1=pd.DataFrame(data=data_knn)
column=data_knn.shape[1]
data_knn_feature=data_knn[:,np.array(range(1,column,2))]
test=pd.DataFrame(data=data_knn_feature)
test.to_csv('KNN_train_P1.csv')
#feature = pd.read_csv('test2.csv', index_col=0)
#data_knn_feature.to_csv('test_knn_feature.csv')

