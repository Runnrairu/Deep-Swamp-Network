# -*- coding: utf-8 -*-

import load_data
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
    W_h2=variable([1000,40000],"W_h2")
    b_h2=variable([40000],"b_h2")
    x_h2=tf.nn.relu(tf.matmul(x_h1, W_h2) + b_h2)
    W_h3=variable([40000,2*(conv+bias)],"W_h3")
    b_h3=variable([2*(conv+bias)],"b_h3")
    param = tf.matmul(x_h2, W_h3) + b_h3
    #ここから分割
    Wb1 = param[0:conv+bias]
    Wb2 = param[conv+bias:2*(conv+bias)]
    W1 = Wb1[0:conv]
    b1 = Wb1[conv:conv+bias]
    W2 = Wb2[0:conv]
    b2 = Wb2[conv:con+bias] 
    
    W1=tf.reshape(W1, [3,3,64,64]) 
    W2=tf.reshape(W2, [3,3,64,64])
    return W1,W2,b1,b2
