import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils as np_utils
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

# 数据预处理
X_train = X_train.reshape(X_train.shape[0],-1)/255
X_test = X_test.reshape(X_test.shape[0],-1)/255
Y_train = np_utils.to_categorical(Y_train,num_classes = 10)
Y_test = np_utils.to_categorical(Y_test,num_classes = 10)

# 创建神经网络
model = Sequential([
    Dense(32,input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

# 定义优化方式
rmsprop = RMSprop(lr = 0.001,rho = 0.9,epsilon = 1e-08,decay= 0.0)

model.compile(
    optimizer = rmsprop,
    loss = 'categorical_crossentropy',# 对数交叉熵
    metrics=['accuracy']
)

print('Training-------------------')
model.fit(X_train,Y_train,nb_epoch=10,batch_size=32)
print('\nTesting------------------')
loss,accuracy = model.evaluate(X_test,Y_test)

print('test loss:',loss)
print('test accuracy:',accuracy)






