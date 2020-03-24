import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 创建一些数据
X = np.linspace(-1,1,200)
np.random.shuffle(X)
Y = 0.5*X + 2 + np.random.normal(0,0.05,(200,))
# 可视化
plt.scatter(X,Y)
plt.show()

X_train,Y_train = X[:160],Y[:160]
X_test,Y_test = X[160:],Y[160:]

# 创建神经网络并搭建隐层
model = Sequential()
model.add(Dense(units = 1,input_dim = 1))

# 选择损失函数
model.compile(loss='mse',optimizer='sgd')

# 训练
print('Training---------------')
costs = []
steps = []
for step in range(301):
    cost = model.train_on_batch(X_train,Y_train)
    print('train cost:',cost)
    costs.append(cost)
    steps.append(step)
# 测试
print('\nTesting----------------')
cost = model.evaluate(X_test,Y_test,batch_size=40)
print('test cost:',cost)
W,b = model.layers[0].get_weights()
print('Weight:',W,'\nbiases:',b)

# 预测
Y_pred = model.predict(X_test)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred)

plt.figure()
plt.plot(steps,costs)
plt.show()
