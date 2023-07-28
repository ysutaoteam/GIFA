
from sklearn.model_selection import train_test_split
import pandas as pd

# 获取目标域的训练集，验证集，测试集
def ParkinsonLoader(path1):
    parkinson_x = pd.read_csv(path1)
    columns = ['Unnamed: 0', ]
    parkinson_x = parkinson_x.drop(columns, axis=1)
    dataset = parkinson_x.values
    parkinson_x = dataset.astype('float32')
    return parkinson_x

x_train=ParkinsonLoader("xtrain.csv")
x_val=ParkinsonLoader("xval.csv")
x_test=ParkinsonLoader("xtest.csv")

y_train=ParkinsonLoader("ytrain.csv")
y_val=ParkinsonLoader("yval.csv")
y_test=ParkinsonLoader("ytest.csv")


from sklearn.preprocessing import StandardScaler, MinMaxScaler
scale = MinMaxScaler()
x_train = scale.fit_transform(x_train)
x_val = scale.transform(x_val)
x_test = scale.transform(x_test)
scale1 = MinMaxScaler()
y_train = scale1.fit_transform(y_train)
y_val = scale1.transform(y_val)
y_test = scale1.transform(y_test)

x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1])
x_val=x_val.reshape(x_val.shape[0],1,x_val.shape[1])
x_test=x_test.reshape(x_test.shape[0],1,x_test.shape[1])

from keras import Sequential
from keras.layers import Bidirectional,LSTM,Dropout,BatchNormalization,TimeDistributed,Dense
from keras.models import Model,load_model
model = Sequential()
input_shape = (None,18)
model.add(Bidirectional(LSTM(units=128,return_sequences=True),input_shape=input_shape))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(units=64,return_sequences=True)))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(units=32,return_sequences=True)))
model.add(Dropout(0.5))

model.add(BatchNormalization(name="lll"))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))     # sigmoid, tanh, relu
# model.add(Dense(1, activation='sigmoid'))

batch_size = 32
model.compile(loss='mse',
              optimizer='adam',
              metrics=['mae'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1000,
          validation_data=(x_val, y_val),
          verbose=1)

dense = Model(inputs=model.input, outputs=model.get_layer('lll').output)
dense.save("sharing_layer2.h5")

dense1 = Model(inputs=model.input, outputs=model.output)
dense1.save("pre2.h5")
model2=load_model('pre2.h5')
dataset_pred1=model2.predict(x_test)
print(dataset_pred1.shape)       # (1191, 1, 1)
dataset_pred1=dataset_pred1.reshape((-1,1))

dataset_pred = scale1.inverse_transform(dataset_pred1)
y_test = scale1.inverse_transform(y_test)

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_test, dataset_pred))
mae = mean_absolute_error(y_test, dataset_pred)
print("rmse:", rmse)
print("mae:", mae)














