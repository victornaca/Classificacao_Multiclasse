import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

#CORREÇÃO DA BASE DE DADOS PARA ENCODER
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

classificador = Sequential()
# NEURONIOS DE CAMADAS OCULTAS
classificador.add(Dense(units = 8, activation = 'tanh', input_dim = 4))
classificador.add(Dense(units = 8, activation = 'tanh'))
# NEURONIOS DE SAIDA
classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer = 'adam',
                      loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])

#INICIALIZAÇÃO DO TREINAMENTO
classificador.fit(previsores, 
                  classe_dummy, 
                  batch_size = 5, 
                  epochs = 100)

#EXPORTAR
classificador_json = classificador.to_json()
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)
#EXPORTAR PARAMETROS
classificador.save_weights('classificador_iris.h5')

