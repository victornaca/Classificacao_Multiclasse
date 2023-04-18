import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

#CORREÇÃO DA BASE DE DADOS PARA ENCODER
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criarRede(optimizer, loos, activation, neurons):
    classificador = Sequential()
    # NEURONIOS DE CAMADAS OCULTAS
    classificador.add(Dense(units = neurons, activation = activation, input_dim = 4))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    # NEURONIOS DE SAIDA
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = optimizer,
                          loss = loos,
                          metrics = ['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size':[5,10],
              'epochs':[100,200],
              'optimizer':['adam','sgd'],
              'loos':['categorical_crossentropy', 'hinge'],
              'activation':['relu', 'tanh'],
              'neurons':[8, 4]}

#INICIALIZAÇÃO DO TREINAMENTO
grid_search = GridSearchCV(estimator=classificador, 
                           param_grid=parametros,
                           scoring='accuracy',
                           cv=5)
grid_search=grid_search.fit(previsores,classe)
melhores_parametros=grid_search.best_params_
melhor_precisao=grid_search.best_score_

