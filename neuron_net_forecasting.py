# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:22:44 2018

@author: tz14
"""

import tensorflow as tf
import numpy as np
import pandas as pd


df = pd.read_csv('train.csv')
x_data=df.drop(['quantity'],axis=1)
noise = np.random.normal(0, 0.05, x_data.shape)
y_data=df['quantity']
y_data=pd.DataFrame({'quantity':y_data})

xs = tf.placeholder(tf.float32,[None,37])
ys = tf.placeholder(tf.float32,[None,1])


# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([37, 50]))
biases_L1 = tf.Variable(tf.zeros([1, 50])+0.1)
Wx_plus_b_L1 = tf.matmul(xs, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

Weights_L2 = tf.Variable(tf.random_normal([50, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1])+0.1)
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#Weights_L3 = tf.Variable(tf.random_normal([50, 1]))
#biases_L3 = tf.Variable(tf.zeros([1, 1])+0.1)
#Wx_plus_b_L3 = tf.matmul(L2, Weights_L3) + biases_L3
#prediction = tf.nn.tanh(Wx_plus_b_L3)

#Weights_L4 = tf.Variable(tf.random_normal([50, 1]))
#biases_L4 = tf.Variable(tf.zeros([1, 1]))
#Wx_plus_b_L4 = tf.matmul(L3, Weights_L4) + biases_L4
#prediction = tf.nn.tanh(Wx_plus_b_L4)

#损失函数
loss = tf.reduce_mean(tf.square(ys - prediction))

# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
a=0
for i in range(500000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 100 == 0:
        b = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
        print(b)
        print(i)
        
print(Weights_L1.value(),biases_L1)
b = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
prediction_value = sess.run(prediction, feed_dict={xs: x_data})
r2 = 1-b/np.average((np.square(y_data-np.average(y_data))))
print(r2)

    # 获得预测值
    

df_test = pd.read_csv('test.csv')
x_data_test=df_test.drop(['quantity'],axis=1)
y_data_test=df_test['quantity']
y_data_test=pd.DataFrame({'quantity':y_data_test})
prediction_value = sess.run(prediction, feed_dict={xs: x_data_test})
r2 = 1-np.average((np.square(prediction_value-y_data_test)))/np.average((np.square(y_data_test-np.average(y_data_test))))

print(r2)
np.savetxt("result.csv", prediction_value, delimiter=",")