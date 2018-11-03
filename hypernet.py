# -*- coding: utf-8 -*-


import tensorflow as tf

def variable(shape,var_name):
    with tf.variable_scope('scope',reuse=tf.AUTO_REUSE):
        
        initial = tf.truncated_normal_initializer( stddev=0.1)
        var = tf.get_variable(name=var_name,shape=shape,initializer = initial)
    return var


d= 32*32*64

conv = 3*3*64*64
bias = 64

def hypernet(t):
    
    W_h1=variable([1,1000],"W_h1")
    b_h1=variable([1000],"b_h1")
    x_h1=tf.nn.relu(tf.matmul(t, W_h1) + b_h1)
    W_h2=variable([1000,10000],"W_h2")
    b_h2=variable([10000],"b_h2")
    x_h2=tf.nn.relu(tf.matmul(x_h1, W_h2) + b_h2)
    W_h3=variable([10000,conv+bias],"W_h3")
    b_h3=variable([conv+bias],"b_h3")
    param = tf.matmul(x_h2, W_h3) + b_h3
    #ここから分割
    W = param[0:conv]
    b= param[conv:conv+bias]
    W=tf.reshape(W, [3,3,64,64]) 
    return W,b







