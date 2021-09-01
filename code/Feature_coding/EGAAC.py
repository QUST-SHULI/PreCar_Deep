import re
from collections import Counter
import readFasta
import numpy as np
import pandas as pd

def GAAC(fastas, **kw):
	group = {
		'alphatic': 'GAVLMI',
		'aromatic': 'FYW',
		'postivecharge': 'KRH',
		'negativecharge': 'DE',
		'uncharge': 'STCPNQ'
	}

	groupKey = group.keys()

	encodings = []
	header = ['#']
	for key in groupKey:
		header.append(key)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		count = Counter(sequence)
		myDict = {}
		for key in groupKey:
			for aa in group[key]:
				myDict[key] = myDict.get(key, 0) + count[aa]

		for key in groupKey:
			code.append(myDict[key]/len(sequence))
		encodings.append(code)

	return encodings

kw=  {'path': r"train_P1.txt",}   
fastas1 = readFasta.readFasta(r"train_P1.txt")

result=GAAC(fastas1, **kw)
data1=np.matrix(result[1:])[:,1:]
data_=pd.DataFrame(data=data1)
data_.to_csv('GAAC_train_P1.csv')