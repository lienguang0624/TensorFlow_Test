import numpy as np
#将列表转换为矩阵
array = np.array([[1,2,3],[2,3,4]])
print(array)

print("矩阵的行数:",array.ndim)

print("矩阵的形状:",array.shape)

print("矩阵的元素数:",array.size)

array = np.array([[1,2,3],[2,3,4]],dtype=np.int)
print("矩阵的数据类型:",array.dtype)
array = np.array([[1,2,3],[2,3,4]],dtype=np.float)
print("矩阵的数据类型:",array.dtype)

a = np.zeros((3,4))
print("生成全部为零的矩阵：\n",a)
a = np.ones((3,4))
print("生成全部为一的矩阵：\n",a)
a = np.arange(10,20,2)
print("生成指定规律的矩阵：\n",a)
a = np.arange(12).reshape((3,4))
print("更改为指定大小的矩阵：\n",a)
a = np.linspace(1,10,6)
print("将一组数自定分为几段：\n",a)

a = np.array([10,20,30,40])
b = np.arange(4)
print("矩阵减法：\n",(b-a))
print("矩阵平方:\n",a**2)
print("矩阵三角函数:\n",np.sin(a))
print("矩阵布尔函数:\n",a>20)

a = np.array([[1,1],[0,1]])
b = np.arange(4).reshape((2,2))
print("a:\n",a,"\nb:\n",b)
print("逐个相乘：\n",a*b)
print("矩阵乘法：\n",np.dot(a,b))
print("矩阵乘法(结果与上面的一致)：\n",a.dot(b))
a = np.random.random((2,4))
print("a矩阵：\n",a)
print("矩阵求转置：\n",a.T)
print("矩阵求平均值：\n",np.mean(a))
print("矩阵求中位数：\n",np.median(a))
print("矩阵按行求和：\n",np.sum(a,axis=1))
print("矩阵按行最小值：\n",np.min(a,axis=1))
print("矩阵按行最大值：\n",np.max(a,axis=1))

A = np.arange(2,14).reshape((3,4))
print("A矩阵：\n",A)
print("对矩阵最小值的索引：\n",np.argmin(A))
print("对矩阵最大值的索引：\n",np.argmax(A))
print("对矩阵进行截断：\n",np.clip(A,5,8))

A = np.arange(3,15).reshape((3,4))
print("A矩阵：\n",A)
print("对A里面的值进行索引：\n",A[2][1],A[:,1])
for column in A.T:
    print("通过迭代对行或列进行操作：\n",column)

A = np.array([1,1,1])
B = np.array([2,2,2])
print("矩阵上下合并：\n",np.vstack((A,B)))
print("矩阵左右合并：\n",np.hstack((A,B)))

A = np.arange(16).reshape((4,4))
print("A矩阵：\n",A)
print("矩阵按行进行分割：\n",np.split(A,2,axis=1))
print("矩阵按列进行分割：\n",np.split(A,2,axis=0))