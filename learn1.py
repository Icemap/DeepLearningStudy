import tensorflow as tf
import numpy as np

# DataCreate
x_data = np.random.random(100).astype(np.float)
y_data = x_data * 0.1 + 0.3

### Create tensorflow structure start ###

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
### Create tensorflow structure stop ###

session = tf.Session()
session.run(init)  # Very important

for step in range(201):
    session.run(train)
    if step % 20 == 0:
        print(step, session.run(Weights), session.run(biases))
