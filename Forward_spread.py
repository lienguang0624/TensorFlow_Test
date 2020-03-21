import tensorflow as tf
import numpy as np
lr = 0.01

w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

(x,y),(x_val,y_val) = tf.keras.datasets.mnist.load_data()
x = 2*tf.convert_to_tensor(x, dtype=tf.float32)/255.-1
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)

x = tf.reshape(x,[-1,28*28])
h1 = x@w1 + tf.broadcast_to(b1,[x.shape[0],256])
h1 = tf.nn.relu(h1)
h2 = h1@w2 +b2
h2 = tf.nn.relu(h2)
out = h2@w3 +b3

loss = tf.squeeze(y-out)
loss = tf.reduce_mean(loss)


grad = tf.GradientTape.gradient(loss,[w1,b1,w2,b2,w3,b3])
w1 = w1.assign_sub(lr * grad[0])
b1 = b1.assign_sub(lr * grad[1])
w2 = w2.assign_sub(lr * grad[2])
b2 = b3.assign_sub(lr * grad[3])
w3 = w3.assign_sub(lr * grad[4])
b3 = b3.assign_sub(lr * grad[5])
print(w1)




# if __name__ == '__main__':
#

