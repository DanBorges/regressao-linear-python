import pandas as pd;
import numpy as np;
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from yellowbrick.classifier import ConfusionMatrix



base = pd.read_csv("insurance.csv", keep_default_na=False) #nesse caso none é uma cas classes, ae para p python entender que não é um valor vazio, faz essa configuração
base.head()
#print(base)
base = base.drop(columns=['Unnamed: 0'])
#print(base.shape)

#print(base.isnull().sum()) exibir se existe algum nulo

y = base.iloc[:,7].values
X = base.drop(base.columns[7], axis=1).values #axies 1 quer dizer que vai excluir a linha e não a coluna
#print(X)

labelEncoder = LabelEncoder()
for i in range(X.shape[1]): # 1 indica que vai percorrer todas as colunas, se fosse 0, ia percorrer todas as linhas
    if(X[:,i].dtype == 'object'):
        X[:,i] = labelEncoder.fit_transform(X[:,i])  # Faz a transformação de dados categóricos em dados numéricos

#print (X)




