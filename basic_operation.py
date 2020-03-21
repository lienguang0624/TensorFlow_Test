import tensorflow as tf
import numpy as np

'''
数值类型
'''
a = 1.2 #python语言方式创建标量
aa = tf.constant(1.2)#TF方式创建标量
print(type(a),type(aa),tf.is_tensor(aa))
x = tf.constant([1,2.,3.3])
print(x)
x = x.numpy()#将张量转换为list
print(type(x))
a = tf.constant([1.2])#创建一个元素的变量
print(a,a.shape)
a = tf.constant([1,2.,3.3])#创建三个元素的变量
print(a,a.shape)
a = tf.constant([[1,2],[3,4]])#创建两行两列的矩阵(constant的形参必须是list格式)
print(a,a.shape)
a = tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])#创建三维张量
print(a,a.shape)

'''
字符串类型
'''
a = tf.constant('Hello')
b = tf.strings.lower(a)#全都转换为小写
print(b)

'''
布尔类型
'''
a = tf.constant(True)#创建布尔类型标量
print(a,a.shape)
a = tf.constant([True , False])#创建布尔类型向量
print(a,a.shape)

'''
数值精度
'''
a = tf.constant(123456789,dtype=tf.int16)#发生溢出
print(a)
a = tf.constant(123456789,dtype=tf.int32)#没有发生溢出
print(a)
a = tf.constant(np.pi,dtype=tf.float32)
print(a)
a = tf.constant(np.pi,dtype=tf.float64)#实现更高精度
print(a)

'''
读取精度
'''
print('before',a.dtype)# 读取精度
if a.dtype != tf.float32:
    a = tf.cast(a,tf.float32)# 精度转换
print('after',a.dtype)

'''
类型转换
'''
a = tf.constant(np.pi,tf.float32)
a = tf.cast(a,tf.double)#转换为高精度
print(a)
a = tf.constant(123456789,tf.int32)
a = tf.cast(a,tf.int16)#转换为低精度(溢出隐患)
print(a)
a = tf.constant([True,False])#布尔类型转换
a = tf.cast(a,tf.int16)
print(a)
a = tf.constant([0,1,2])# 除0外都是True
a = tf.cast(a,tf.bool)
print(a)

'''
待优化张量
'''
a = tf.constant([-1,0,1,2])
aa = tf.Variable(a)#普通张量转换为待优化张量
print(aa.name,aa.trainable)
a = tf.Variable([[1,2],[3,4]])
print(a)

'''
创建张量
'''
a = tf.convert_to_tensor([1,2.])#从List中创建张量
print(a)
a = tf.convert_to_tensor(np.array([[1,2.],[3,4]]))#从Numpy中创建张量
print(a)
a = tf.zeros([])#创建全0或全1的标量
print(a)
a = tf.ones([])
print(a)
a = tf.zeros([1])#创建全0或全1的向量
print(a)
a = tf.ones([1])
print(a)
a = tf.zeros([2,2])#创建全0或全1的张量
print(a)
a = tf.ones([2,2])
print(a)
a = tf.zeros_like(a)#copy一个张量，并给定0或给定1
print(a)
a = tf.fill([4,4],99)#设定一个标量，并给予指定数值
print(a)
a = tf.random.normal([2,2],mean=1,stddev=2)#创建自定义均值和标准差的正态分布
print(a)
a = tf.random.uniform([2,2],maxval=100,dtype=tf.int32)#创建自定义的均值分布
print(a)
a = tf.range(1,10,delta=2)#创建等差数列（序列）
print(a)
'''
张量的典型应用
'''
out = tf.random.uniform([4,10])#随机模拟网络输出
y = tf.constant([2,3,2,0])#随机构造样本真实标签
y = tf.one_hot(y,depth=10)#one-hot编码
loss = tf.keras.losses.mse(y,out)#计算每个样本的mse
loss = tf.reduce_mean(loss)#平均mse
print(loss)
z = tf.random.normal([4,2])#张量加法演示
b = tf.zeros([2])
z = z + b
print(z)
x = tf.random.uniform([2,4])#张量乘法演示
w = tf.ones([4,3])
b = tf.zeros([3])
o = x@w + b
print(o)
'''
三维张量
'''
(x_train,y_train),(x_test,y_test) =tf.keras.datasets.imdb.load_data(num_words=10000)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=80)
print(x_train.shape)
embedding = tf.keras.layers.Embedding(10000,100)
out = embedding(x_train)
print(out.shape)
'''
四维张量
'''
x = tf.random.normal([4,32,32,3])
layer = tf.keras.layers.Conv2D(16,kernel_size=3)
out = layer(x)
print(out.shape)
'''
索引
'''
x = tf.random.normal([4,32,32,3])
print('索引',x[0][1]),x[1,2,3]
'''
切片
'''
print('切片',x[1:3])
'''
改变视图
'''
x = tf.range(96)
x = tf.reshape(x,[2,4,4,3])
print(x)
x = tf.reshape(x,[2,-1,3])
print(x)

'''
增减维度
'''
x = tf.random.uniform([28,28],maxval=10,dtype=tf.int32)
x = tf.expand_dims(x,axis=2)
x = tf.expand_dims(x,axis=0)
print(x,x.shape)
x = tf.squeeze(x,axis=0)
x = tf.squeeze(x,axis=2)
print(x,x.shape)
'''
交换维度
'''
x = tf.random.normal([2,32,32,3])
x = tf.transpose(x,perm=[0,3,1,2])
print(x)
'''
复制数据
'''
x = tf.constant([1,2])
x = tf.expand_dims(x, axis=0)
x = tf.expand_dims(x, axis=0)
print(x)
x = tf.tile(x, multiples=[2,2,1])
print(x)
'''
数学运算
'''
a = tf.range(5)
print(a)
b = tf.constant(2)
a = a//b#整除运算
print(a)
a = a%b#余除运算
print(a)
a = tf.range(5.)
a = a**2#乘方运算
print(a)
a = a**0.5
print(a)
a = 2**a#指数运算
print(a)
a = tf.math.log(a)#对数运算
print(a)
a = tf.random.normal([4,3,28,32])#矩阵乘法
b = tf.random.normal([4,3,32,2])
a = a@b
print(a)