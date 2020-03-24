import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras.models import load_model

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

X_train = X_train.reshape(-1,1,28,28)
X_test = X_test.reshape(-1,1,28,28)
Y_train = np_utils.to_categorical(Y_train,num_classes = 10)
Y_test = np_utils.to_categorical(Y_test,num_classes = 10)

'''
设定卷积神经网路格式
'''
model = Sequential()
model.add(Convolution2D(
    nb_filter = 32,
    nb_row = 5,
    nb_col = 5,
    border_mode = 'same',
    input_shape=(1,28,28)
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size = (2,2),
    strides=(2,2),
    border_mode = 'same',
))
model.add(Convolution2D(
    nb_filter = 64,
    nb_row = 5,
    nb_col = 5,
    border_mode = 'same'
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size = (2,2),
    strides=(2,2),
    border_mode = 'same',
))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
'''
设定学习方法
'''
adam = Adam(lr = 1e-4)
'''
创建模型
'''
model.compile(
    optimizer = adam,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
'''
训练
'''
print('Training-------------------')
model.fit(X_train,Y_train,nb_epoch=2
          ,batch_size=32)
'''
测试
'''
print('\nTexting-------------------------')
loss,accuracy = model.evaluate(X_test,Y_test)

print('\ntext loss:',loss)
print('\ntest accuracy:',accuracy)

'''
保存
'''
print('test before save:',model.predict((X_test[0:2])))
model.save('my_model.h5')

'''
读取
'''
model = load_model('my_model.h5')
print('\ntest after losd:',model.predict(X_test[0:2]))