# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

T = 1

d=32*32*64
def p(t):
    p_T = 0.4
    return 1-t/T*(1-p_T)

def G(t):#Fukasawa_schemeで使用
    return 1.0

def G_nm(n,t_now):
    return float(1)/(G(t)*n)


def tW_def(n,task_name):
    if task_name == "Fukasawa_scheme":
        t,W = Fukasawa_scheme(n,T)
    elif task_name == "Simplicity_scheme":
        t,W = Simplicity_scheme(n,T)
    elif task_name == "Euler_Maruyama_scheme" or task_name == "Milstein_scheme":
        t,W = Euler_Maruyama_scheme(n,T)
    elif task_name == "ODEnet":
        t,W = ODEnet(n,T)
    else:
        print("Invarid!")
    return t,W





def Fukasawa_scheme(n,T):#今回最も特殊なスキーム
    
    t = [0]*(n)
    W=[0]*(n)
    t_now=0
    m=0
    a= pow(1+2.0/d,1+d/2.0)
    
    while(t_now < (T-G_nm(n,t_now))):
        N = np.random.normal()
        E = np.random.exponential()
        ab_N = np.absolute(N)
        Z = ab_N*ab_N+2*E/d
        G__nm = G_nm(t_now)
        delta_t = G__nm*np.exp(-Z)
        t[n]= delta_t
        t_now +=delta_t
        W[n] = pow(G__nm*a*d*Z*np.exp(-Z),0.5)*N/ab_N
        m+=1
    if m<n:
        delta_euler_t = (T-t_now)/(n-m) 
        sigma_euler_t = pow(delta_euler_t,0.5)
        for i in range(n-m):
            t[m+i] = delta_euler_t
            W[m+i] = np.random.normal(0,sigma_euler_t) 
    return t,W

def ODEnet(n,T):#先行研究
    delta_t = float(T)/n
    t = [delta_t]*(n)
    W = [0]*(n)
    return t,W





def Simplicity_scheme(n,T):
    delta_t = float(T)/n
    t = [delta_t]*(n)
    sigma = pow(delta_t,0.5)
    W = np.random.choice([-sigma,sigma], n, replace=True)
    
    
    return t,W



def Euler_Maruyama_scheme(n,T):
    delta_t = float(T)/n
    t = [delta_t]*(n)
    sigma = pow(delta_t,0.5)
    W = np.random.normal(0,sigma,n)
    return t,W

    
    
    

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def variable(shape,var_name):
    with tf.variable_scope('scope',reuse=tf.AUTO_REUSE):
        initial = tf.truncated_normal_initializer(shape, stddev=0.1)
        var = tf.get_variable(name=var_name,shape,initializer = initial)
    return var



def SDE_model(X,t,W,task_name):
    
    depth = len(t) 
    
    W_conv = weight_variable([5, 5, 3, 64],"W_conv")
    b_conv = bias_variable([64],"b_conv")
    X_image = tf.reshape(X, [-1,32,32,3])
    
    X_image = conv2d(x, W_conv, strides=[1, 1, 1, 1], padding='SAME')
    t_now = 0
    for i in range(depth):
        delta_t = t[i]
        delta_W = W[i]
        X_image = Res_flow(X_image,t_now,delta_t,delta_W,task_name)
        t_now += delta_t
    
    
    # 最大値プーリング(平均値のほうがよくない？)
    X_pool = tf.nn.max_pool(X_image, ksize=[1, 4, 4, 1],strides=[1, 4, 4, 1])
    
    # 全結合層
    W_fc1 = weight_variable([8* 8 * 64,4096],"w_fc1")# ここの7はちゃんとプーリング後の大きさを正しく計算する。
    b_fc1 = bias_variable([4096],"b_fc1")
    X_pool_flat = tf.reshape(X_pool, [-1,  8* 8 * 64])#同じく
    X_fc1 = tf.nn.relu(tf.matmul(X_pool2_flat, W_fc1) + b_fc1)


    # 出力層　　　　　　　　　
    W_fc2 = weight_variable([4096, 10],"W_fc2")
    b_fc2 = bias_variable([10],"b_fc2")
    y_conv = tf.matmul(X_fc1, W_fc2) + b_fc2

    return y_conv #メインではこれがnetという名前になる 
    
    
    


def ResFlow(inpt,t_now,delta_t,delta_w,task_name):
    
    f_x = Res_func(inpt,task_name)
    p_t = p(t)
    
    if task_name == "Milstein_scheme":
        return inpt+p_t*delta_t*f_x +np.pow(p_t*(1-p_t),0.5)*delta_w*f_x+()*(np.pow(delta_w,2)-delta_t)#ミルシュタインスキーム特有のやつ
    else:
        return inpt+p_t*delta_t*f_x +np.pow(p_t*(1-p_t),0.5)*delta_w*f_x
   
def Res_func(inpt,task_name):
    W_conv1 = variable([3, 3, 64, 64],"W_conv1")
    b_conv1 = variable([64],"b_conv1")
    W_conv2 = variable([3, 3, 64, 64],"W_conv2")
    b_conv2 = variable([64],"b_conv2")
    
    
    inpt_ = tf.nn.relu(conv2d(inpt, W_conv1)+b_conv1)#バッチ正規化したい
    output = conv2d(inpt, W_conv2)+b_conv2 #ここReluかますかは迷いどころ
    
    
    
    return output
