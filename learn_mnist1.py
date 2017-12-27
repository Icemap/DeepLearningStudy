import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import subprocess

myFont = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')  # fontFamily

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])   # None表示这个张量可以是任意大小的
Weight = tf.Variable(tf.zeros([784, 10]))     # 张量1是图，张量2是Tag
biases = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, Weight) + biases)
y_ = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(y_ * tf.log(y))  # 交叉熵计算，评估模型当前的改进方向，以便继续学习

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

session = tf.Session()
init = tf.initialize_all_variables()

# 前面都是懒加载。下面run函数一次性注入到C/C++源码。session是实时的加载，调起了C/C++
losses = []
indexs = []
plt.ion()
plt.figure(1)
plt.title("交叉熵展示", fontproperties=myFont)
plt.xlabel("迭代次数", fontproperties=myFont)
plt.ylabel("交叉熵值", fontproperties=myFont)
session.run(init)

for i in range(1000):
    part_xs, part_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: part_xs, y_: part_ys})

    if i % 10 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct_rate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(session.run(correct_rate, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        losses.append(session.run(loss, feed_dict={x: part_xs, y_: part_ys}))
        indexs.append(i)
        plt.plot(indexs, losses, 'mx-')
        plt.draw()
        plt.pause(0.1)

plt.savefig('./minst1.png')
