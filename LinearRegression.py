import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# loading the data points
data = pd.read_csv('Salary_Data.csv')
X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values

# Running linear regression
x = tf.placeholder("float")
y = tf.placeholder("float")
W = tf.Variable(initial_value=np.random.randn(),name="W")
b = tf.Variable(initial_value=np.random.randn(),name="b")

Y_pred = W*X + b

loss = tf.reduce_mean((y-Y_pred)*(y-Y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1000):
        for (_x,_y) in zip(X,Y):

            sess.run(optimizer,feed_dict={x:_x,y:_y})

    print(sess.run(W))
# plotting the data points and result
    plt.scatter(X,Y)
    plt.plot(X,sess.run(W)*X+sess.run(b))
    plt.show()