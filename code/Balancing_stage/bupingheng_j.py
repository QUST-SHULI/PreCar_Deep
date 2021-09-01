import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import BorderlineSMOTE



data_=pd.read_csv(r'train_P1.csv')
data1=np.array(data_)
data=data1[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((114,1))#Value can be changed   int(m1/2)
label2=np.zeros((716,1))
#label1=np.ones((int(m1/2),1))#Value can be changed
#label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)


X=shu
y=label#.astype('int64')


bor_smo = BorderlineSMOTE(kind='borderline-1',sampling_strategy={0: 716,1:716},random_state=42,k_neighbors=4) #kind='borderline-2'
X_resampled, y_resampled = bor_smo.fit_sample(X, y)

shu=X_resampled
X1=scale(shu)
y1=y_resampled

#shu2 =X_resampled
#shu3 =y_resampled
data_csv = pd.DataFrame(data=X1)
data_csv.to_csv('SMOTETomek_train_feature_GL.csv')
data_csv = pd.DataFrame(data=y1)
data_csv.to_csv('label_SMOTETomek_train_feature_GL.csv')