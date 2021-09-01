import numpy as np
import pandas as pd
#import xgboost as xgb
import scipy.io as sio
import utils.tools as utils
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
# from HH import to_categorical,categorical_probas_to_classes,calculate_performace
from sklearn.preprocessing import scale,StandardScaler
from keras.layers import Dense, merge,Input,Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Convolution2D, AveragePooling2D, MaxPooling2D,BatchNormalization,GlobalMaxPooling2D
from keras.layers import Embedding, Bidirectional, LSTM, Conv1D
from keras.layers import AveragePooling1D,MaxPooling1D
# from keras_contrib.layers.crf import crf
# from keras_contrib.layers import CRF
from keras.layers import Flatten 
# import utils.tools as calculate_performace
# import utils.tools as categorical_probas_to_classes
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from keras.optimizers import RMSprop
import math 
# from fscore_callback import fscore_callback
def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)
def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y
def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp + 1e-06)
    npv = float(tn)/(tn + fn + 1e-06)
    sensitivity = float(tp)/ (tp + fn + 1e-06)
    specificity = float(tn)/(tn + fp + 1e-06)
    mcc = float(tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-06)
    f1=float(tp*2)/(tp*2+fp+fn+1e-06)
    return acc, precision,npv, sensitivity, specificity, mcc, f1

def get_shuffle(dataset,label):    
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label 
data_=pd.read_csv(r'A.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))#Value can be changed
label2=np.zeros((int(m1/2),1))
#label1=np.ones((116,1))#Value can be changed
#label2=np.zeros((702,1))
label=np.append(label1,label2)

X_=scale(data)
y_= label
X,y=get_shuffle(X_,y_)
sepscores = []
sepscores_ = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5

[sample_num,input_dim]=np.shape(X)
out_dim=2
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
probas_cnn=[]
tprs_cnn = []
sepscore_cnn = []

skf= StratifiedKFold(n_splits=10)

def get_CNNLSTM_model(input_dim,out_dim):  
    model = Sequential()
    #model.add(Conv1D(64, 1,input_shape=(1,511)))
    model.add(filters = 64, kernel_size = 3, padding = 'same', activation= 'relu')
    model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))
    model.add(Conv1D(32, 1))
    model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))
    model.add(Bidirectional(LSTM(int(input_dim/2), return_sequences=True)))
    model.add(Bidirectional(LSTM(int(input_dim/4), return_sequences=True)))
    #model.add(Bidirectional(LSTM(64 // 2, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(int(input_dim), activation = 'relu'))    
    model.add(Dense(int(input_dim/4), activation = 'relu'))
    model.add(Dense(out_dim, activation = 'softmax',name="Dense_2"))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])
    #sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
    return model


for train, test in skf.split(X,y): 
    y_train=to_categorical(y[train])#generate the resonable results
    # y_train = np.expand_dims(y_train,2)
    
    X_train=np.reshape(X[train],(-1,1,input_dim))
    X_test=np.reshape(X[test],(-1,1,input_dim))
    
    cv_clf =get_CNNLSTM_model(input_dim,out_dim)
    hist=cv_clf.fit(X_train, y_train,batch_size=32,nb_epoch=20)
    y_test=to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]       
    y_score=cv_clf.predict(X_test)#the output of  probability
    yscore=np.vstack((yscore,y_score))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= categorical_probas_to_classes(y_score)
    acc, precision,npv, sensitivity, specificity, mcc,f1 = calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('CBLM:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
    hist=[]
    cv_clf=[]
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores
row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('1yscore_CNN_BiLSTM20_A.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('1ytest_CNN_BiLSTM20_A.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='CBLM ROC (area = %0.2f%%)' % auc_score)
