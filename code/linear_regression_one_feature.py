import numpy as np
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants  

# Model linear regression y = Wx + b
x = tf.placeholder(tf.float32, [None, 1], name="x")
W = tf.Variable(tf.zeros([1,1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")
product = tf.matmul(x,W)
y = tf.add(product, b, name="y")
y_ = tf.placeholder(tf.float32, [None, 1], name="y_")

# Cost function sum((y_-y)**2)
cost = tf.reduce_mean(tf.square(y_-y), name="cost")

# Training using Gradient Descent to minimize cost
train_step = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
steps = 1000
for i in range(steps):
  # Create fake data for y = W.x + b where W = 2, b = 0
  xs = np.array([[i]])
  ys = np.array([[2*i]])
  # Train
  feed = { x: xs, y_: ys }
  sess.run(train_step, feed_dict=feed)
  print("After %d iteration:" % i)
  print("W: %f" % sess.run(W))
  print("b: %f" % sess.run(b))
  # Suggested by @jihobak
  print("cost: %f" % sess.run(cost, feed_dict=feed))

# NOTE: W should be close to 2, and b should be close to 0
feed = { x: [[4],[16]] }
print(sess.run(y, feed_dict=feed))

graph = convert_variables_to_constants(sess, sess.graph_def, ["y"]) #out为保存网络的最后输出节点名称  
tf.train.write_graph(graph, '.', 'graph.pb', as_text=False)  
writer = tf.summary.FileWriter("logs/",sess.graph)
print(tf.global_variables())