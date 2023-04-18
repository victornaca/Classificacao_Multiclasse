import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.utils import np_utils

arquivo = open('classificador_iris.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_iris.h5')

novo = np.array([[1.2, 7, 2.3, 4]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.7)

#SUBIR BASE
base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
#CORREÇÃO DA BASE DE DADOS PARA ENCODER
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

classificador.compile(optimizer = 'adam',
                      loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])

resultado = classificador.evaluate(previsores, classe_dummy)
