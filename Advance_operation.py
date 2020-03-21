import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
# '''
# 拼接(不创建新维度)
# '''
# a = tf.random.normal([4,35,8])
# b = tf.random.normal([6,35,8])
# c = tf.concat([a,b],axis=0)
# print(c)
# a = tf.random.normal([6,35,8])
# b = tf.random.normal([6,35,8])
# c = tf.concat([a,b],axis=2)#不同维度进行拼接
# print(c)
#
# '''
# 堆叠(创建新维度)
# '''
# a = tf.random.normal([35,8])
# b = tf.random.normal([35,8])
# c = tf.stack([a,b],axis=0)
# print(c)
#
# '''
# 分割
# '''
# a = tf.random.normal([4,35,8])
# result = tf.split(a,num_or_size_splits=[1,1,2],axis=0)
# print(result[2])
#
# '''
# 向量范数
# '''
# x = tf.ones([2,2])
# y = tf.norm(x,ord = 1)#绝对值之和
# print(y)
# y = tf.norm(x,ord = 2)#平方和开根号
# print(y)
# y = tf.norm(x,ord = np.inf)#取绝对值中最大值
# print(y)
#
# '''
# 最值，均值，和
# '''
# x = tf.random.normal([4,10])
# y = tf.reduce_max(x,axis=1)#返回选定轴中最大的
# print(y)
# x = tf.random.normal([4,10])
# y = tf.reduce_min(x,axis=1)#返回选定轴中最小的
# print(y)
# x = tf.random.normal([4,10])
# y = tf.reduce_mean(x,axis=1)#返回选定轴的平均值
# print(y)
# x = tf.random.normal([4,10])
# y = tf.reduce_sum(x,axis=1)#返回选定轴的和
# print(y)
# out = tf.random.normal([2,10])
# out = tf.nn.softmax(out,axis=1)
# print(out)
# pred = tf.argmax(out,axis=1)#找出最大的，返回位置
# print(pred)
# '''
# 张量比较
# '''
# out = tf.random.normal([100,10])
# out = tf.nn.softmax(out)
# pred = tf.argmax(out,axis=1)
# print(pred)
# y = tf.random.uniform([100],dtype=tf.int64,maxval=10)
# out = tf.equal(pred,y)
# out = tf.cast(out,dtype=tf.int32)
# correct = tf.reduce_sum(out)
# print(correct)
# '''
# 填充
# '''
# a = tf.constant([1,2,3,4,5,6])
# b = tf.constant([7,8,1,6])
# b = tf.pad(b,[[0,2]])#左边不填充，右边填充两个
# print(b)
# c = tf.stack([a,b],axis=0)#堆叠
# print(c)
# total_words = 10000
# max_review_len = 80
# embedding_len = 100
# (x_train,y_train),(x_test,y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)
# x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_review_len,truncating='post',padding='post')
# x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_review_len,truncating='post',padding='post')
# print(x_train.shape)
#
# x = tf.random.normal([4,28,28,1])
# x = tf.pad(x,[[0,0],[2,2],[2,2,],[0,0]])
# print(x.shape)
#
# '''
# 复制
# '''
# x = tf.random.normal([4,32,32,1])
# x = tf.tile(x,[2,3,3,1])#复制但不形成新的通道
# print(x)
#
# '''
# 数据限幅
# '''
# x = tf.range(9)
# x = tf.maximum(x,2)
# print(x)
# x = tf.minimum(x,7)
# print(x)
# x = tf.minimum(tf.maximum(x,3),6)
# print(x)
# x = tf.clip_by_value(x,4,5)
# print(x)
#
# '''
# 根据索引号收集数据
# '''
# x = tf.random.uniform([4,35,8],maxval=100,dtype=tf.int32)
# y = tf.gather(x,[0,1],axis=0)
# print(y)
# y = tf.gather(x,[0,2,4],axis=2)
# print(y)
# y = tf.gather_nd(x,[[1,1],[2,2],[3,3]])
# print(y)
# '''
# tf.boolean_mask
# '''
# x = tf.boolean_mask(x,mask=[True,False,False,True],axis=0)
# print(x)
# x = tf.boolean_mask(x,mask=[True,False,False,True,True,False,False,True],axis=2)
# print(x)
# a = tf.ones([3,3])
# b = tf.zeros([3,3])
# cond = tf.constant([[True,False,False],[False,True,False],[True,True,False]])
# print(tf.where(cond,a,b))
# '''
# tf.where
# '''
# x = tf.random.normal([3,3])
# mask = x>0
# indices = tf.where(mask)
# print(tf.gather_nd(x,indices))
#
# '''
# scatter_nd 在指定位置写入指定数据
# '''
# indices = tf.constant([[4],[3],[2],[1]])
# updatas = tf.constant([4.4,3.3,1.1,7.7])
# print(tf.scatter_nd(indices,updatas,[8]))
#
# indices = tf.constant([[1],[3]])
# updates = tf.constant([[[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]], [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]])
# print(tf.scatter_nd(indices,updates,[4,4,4]))
#
# '''
# tf.meshgrid
# '''
# points = [] # 保存所有点的坐标列表
# for x in range(-8,8,100): # 循环生成 x 坐标，100 个采样点
#     for y in range(-8,8,100): # 循环生成 y 坐标，100 个采样点
#         z = x + y # 计算每个点(x,y)处的 sinc 函数值
#         points.append([x,y,z]) # 保存采样点
# x = tf.linspace(-8.,8,100)
# y = tf.linspace(-8.,8,100)
# x,y = tf.meshgrid(x,y)
# print(x.shape,y.shape)
# z = tf.sqrt(x**2+y**2)
# z = tf.sin(z)/z
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.contour3D(x.numpy(),y.numpy(),z.numpy(),50)
# plt.show()
'''
经典数据集加载
'''
(x,y),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
train_db = tf.data.Dataset.from_tensor_slices((x,y))
print(train_db)
train_db = train_db.shuffle(10000)
train_db = train_db.batch(128)
print(train_db)
def preprocess(x,y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y

train_db = train_db.map(preprocess)







