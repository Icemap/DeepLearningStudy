import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = input1 * input2

with tf.Session() as session:
    print(session.run(output, feed_dict={input1: [7.], input2: [2.]}))
