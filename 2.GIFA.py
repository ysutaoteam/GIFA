import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from example_MB import example


def ParkinsonLoader(path1):
    parkinson_x = pd.read_csv(path1)
    columns = ['Unnamed: 0', ]
    parkinson_x = parkinson_x.drop(columns, axis=1)
    columns = ['motor_UPDRS']
    parkinson_y = pd.DataFrame(parkinson_x, columns=columns)
    columns = ['motor_UPDRS', 'total_UPDRS']
    parkinson_x = parkinson_x.drop(columns, axis=1)
    dataset = parkinson_x.values
    parkinson_x = dataset.astype('float32')
    dataset = parkinson_y.values
    parkinson_y = dataset.astype('float32')
    x_train, x_test, y_train, y_test = train_test_split(parkinson_x, parkinson_y, test_size=0.2, shuffle=True,random_state=89)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=True)
    return x_train, x_val, x_test, y_train, y_val, y_test

feature_names = ['age', 'sex', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA','NHR','HNR','RPDE','DFA','PPE']
predict_name = 'motor_UPDRS'

rmse1=[]
mae1=[]
r=[]
mape1=[]
for k in range(42):
    p1 = "p"
    p2 = ".csv"
    c = np.hstack((p1, k + 1))
    c = np.hstack((c, p2))
    s = ""
    path = s.join(c)

    print("k=",k)

    x_train, x_val, x_test, y_train, y_val, y_test = ParkinsonLoader(path)

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
    model.add(Bidirectional(LSTM(units=32,return_sequences=True),input_shape=input_shape))  ## True返回所有节点的输出；FALSE返回最后一个节点的输出
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(units=16, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(name="lstm"))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    # model.add(Dense(1, activation='sigmoid'))

    batch_size = 32
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=200,
              validation_data=(x_test, y_test),
              verbose=0)

    dense = Model(inputs=model.input, outputs=model.get_layer('lstm').output)
    dense_out = dense.predict(x_train)
    train_out = dense_out.reshape((dense_out.shape[0], dense_out.shape[2]))

    dense_out1 = dense.predict(x_test)
    test_out = dense_out1.reshape((dense_out1.shape[0], dense_out1.shape[2]))

    layer=load_model('sharing_layer2.h5')
    share=layer.predict(x_train)
    train_share=share.reshape((share.shape[0],share.shape[2]))
    print(share.shape)
    share1=layer.predict(x_test)
    test_share=share1.reshape((share1.shape[0],share1.shape[2]))

    train = np.hstack((train_share, y_train))

    all = pd.DataFrame(train)
    all.to_csv("all_feature_FC.csv")

    data = pd.read_csv("all_feature_FC.csv")
    columns = ['Unnamed: 0', ]
    data = data.drop(columns, axis=1)
    # print(data.shape)  # (86, 65)

    method = 'IAMB'        # IAMB
    K_flag = False
    if method == "KIAMB":
        K = float(input("k: "))
        K_flag = True
    elif method == "FBEDk":
        # K = int(input("k: "))
        K = 10
        K_flag = True

    list_target = [64]

    alpha = 0.05
    isdiscrete = 0
    if isdiscrete == "1":
        isdiscrete = True
    elif isdiscrete == "0":
        isdiscrete = False

    if K_flag:
        MB = example(method, data, list_target, alpha, isdiscrete, K)
    else:
        MB = example(method, data, list_target, alpha, isdiscrete)

    train_share = train_share[:, MB]
    test_share = test_share[:, MB]


    def new(share, out):
        new1 = []
        for i in range(share.shape[0]):
            new = np.hstack((share[i, :], out[i, :]))
            new1.extend(new)
        new_feature = np.array(new1).reshape((share.shape[0], -1))
        print(new_feature.shape)
        return new_feature


    train1 = new(train_share, train_out)
    test1 = new(test_share, test_out)

    model1 = Sequential()
    model1.add(Dense(input_dim=train1.shape[1], units=32, activation='relu'))
    model1.add(Dropout(0.1))
    model1.add(Dense(units=16, activation='relu'))
    model1.add(Dropout(0.1))
    model1.add(Dense(units=8, activation='relu'))
    model1.add(Dropout(0.5))
    model1.add(Dense(units=1, activation='sigmoid'))

    batch_size = 16
    model1.compile(loss='mse',optimizer='adam', metrics=['mae'])
    model1.fit(train1, y_train,batch_size=batch_size,epochs=100,verbose=0)
    dense_out = model1.predict(test1)

    testPredict = scale1.inverse_transform(dense_out)
    testY = scale1.inverse_transform(y_test)

    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(testY, testPredict))
    mae = mean_absolute_error(testY, testPredict)

    rmse1.append(rmse)
    mae1.append(mae)

print("rmse:", np.mean(rmse1))
print("mae:", np.mean(mae1))













