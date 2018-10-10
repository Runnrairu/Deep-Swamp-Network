# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np



def tW_def(n,name):
    if name == "Fukasawa_scheme":
        t,W = Fukasawa_scheme(1)
    elif name == "Simplicity_scheme":
        t,W = Simplicity_scheme(1)
    elif name == "Euler_Maruyama_scheme":
        t,W = Euler_Maruyama_scheme(n,1)
    elif name == "Milstein_scheme":
        t,W == Milstein_scheme(n,1)
    else:
        print("Invarid!")
    return t,W

def Fukasawa_scheme(T):
    
    return t,W

def Simplicity_scheme(T):
    
    return t,W




def Euler_Maruyama_scheme(n,T):
    
    return t,W


def Milstein_scheme(n,T):
    
    retrun t,W


    
def SDE_model(X,t,W):
    
    depth = len(t) #shape取得関数を調べてここで使う
    
    W_conv = weight_variable([5, 5, 3, 64])
    b_conv = bias_variable([64])
    X_image = tf.reshape(X, [-1,32,32,1])
    
    X_image = tf.nn.conv2d(x, W_conv, strides=[1, 1, 1, 1], padding='SAME')
    t_now = 0
    for i in range(depth):
        delta_t = t[i]
        delta_W = W[i]
        X_image = Res_flow(X_image,t_now,delta_t,delta_W)
        t_now += delta_t
    
    
    # 平均値プーリング
    
    
    # 全結合層
    
    
    return net 
    
    
    


def ResFlow(inpt,t_now,delta_t,delta_w):
    
    f_x = Res_func(inpt)
    p_t = p(t)
    
    return inpt+p_t*delta_t*f_x +np.pow(p_t*(1-p_t),0.5)*delta_w*f_x
   
def Res_func(inpt):
    W_conv1 = weight_variable([3, 3, 64, 64])
    b_conv1 = bias_variable([64])
    W_conv2 = weight_variable([3, 3, 64, 64])
    b_conv2 = bias_variable([64])
   
   
   
   
   return output
