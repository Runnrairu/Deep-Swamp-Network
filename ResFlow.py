# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

T = 1.0

d=32*32*64



def p(t):
    p_T = 0.5

    return 1-t/T*(1-p_T)

def G(t):#Fukasawa_schemeで使用
    return 1.0

def G_nm(n,t_now):
    return 1.0/(G(t_now)*n)




def tW_def(n,task_name):
    t=[]
    W=[]
    if task_name == "Fukasawa_scheme":
        t,W = Fukasawa_scheme(n,T)
    elif task_name == "Simplicity_scheme":
        t,W = Simplicity_scheme(n,T)
    elif task_name == "Euler_Maruyama_scheme" or task_name == "Milstein_scheme":
        t,W = Euler_Maruyama_scheme(n,T)
    elif task_name == "ODEnet" or task_name=="test" :
        t,W = ODEnet(n,T)
    else:
        print("Invarid!")
    return t,W




def Fukasawa_scheme(n,T):#今回最も特殊なスキーム

    t = [0]*(n+1)
    W=[0]*(n+1)
    N_list=np.random.normal(0,1.0,[n+1])
    E_list=np.random.exponential(1.0,[n+1])
    t_now=0
    m=0
    d=1
    a= np.power(1+2.0/d,1+d/2.0)

    while(t_now < (T-a*G_nm(n,t_now)) and m<n):

        N = N_list[m]
        E = E_list[m]
        ab_N = np.absolute(N)
        Z = (ab_N*ab_N+2*E)/d
        G__nm = G_nm(n,t_now)
        delta_t = G__nm*a*np.exp(-Z)
        t[m]= delta_t
        t_now += delta_t
        #if t_now>T:

        #    t_now -= delta_t

        #    break
        W[m] = np.power(G__nm*a*d*Z*np.exp(-Z),0.5)*N/ab_N
        m+=1

    delta_euler_t = (T-t_now)/(n-m+1)
    sigma_euler_t = np.power(delta_euler_t,0.5)
    for i in range(n-m+1):


        t[m+i] = delta_euler_t
        W[m+i] = sigma_euler_t*N_list[m+i]

    return t,W

def ODEnet(n,T):#先行研究
    delta_t = float(T)/(n+1)
    t = [delta_t]*(n+1)
    W = [0]*(n+1)
    return t,W





def Simplicity_scheme(n,T):
    delta_t = float(T)/(n+1)
    t = [delta_t]*(n+1)
    sigma = np.power(delta_t,0.5)
    W = np.random.choice([-sigma,sigma], n+1, replace=True)


    return t,W



def Euler_Maruyama_scheme(n,T):
    delta_t = float(T)/(n+1)
    t = [delta_t]*(n+1)
    sigma = np.power(delta_t,0.5)
    W = np.random.normal(0,sigma,n+1)
    return t,W





def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def variable(shape,var_name,Flow=False):

    v=0.01
    long = len(shape)

    with tf.variable_scope('scope',reuse=tf.AUTO_REUSE):
        if long == 1:
            initial = tf.constant(0.0,shape=shape,dtype="float32")
            var = tf.get_variable(name=var_name,initializer = initial)
        else:
            initial = tf.truncated_normal_initializer(stddev=v)
            var = tf.get_variable(name=var_name,shape=shape,initializer = initial)
    return var

Z_imagetest = []

def SDE_model(X,t,W,task_name_tr):
    global Z_imagetest
    Z_imagetest=[]
    depth =52

    W_conv = variable([5, 5, 3, 64],"W_conv")
    b_conv = variable([64],"b_conv")
    X_image = tf.reshape(X, [-1,32,32,3])

    X_image = conv2d(X_image, W_conv)
    t_now = 0

    for i in range(depth):
        delta_t = t[i]
        delta_W = W[i]
        t_now += delta_t
        X_image = Res_flow(X_image,t_now,delta_t,delta_W,task_name_tr)
        #X_image=tf.Print(X_image,[X_image])




    # 最大値プーリング(平均値のほうがよくない？)
    X_pool = tf.nn.max_pool(X_image, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding = "VALID")

    # 全結合層
    W_fc1 = variable([16* 16 * 64,4096],"W_fc1")# ここの7はちゃんとプーリング後の大きさを正しく計算する。
    b_fc1 = variable([4096],"b_fc1")
    X_pool_flat = tf.reshape(X_pool, [-1,  16* 16 * 64])#同じく
    X_fc1 = tf.nn.relu(tf.matmul(X_pool_flat, W_fc1) + b_fc1)

    W_fc2 = variable([4096, 4096],"W_fc2")
    b_fc2 = variable([4096],"b_fc2")
    X_fc2 = tf.matmul(X_fc1, W_fc2) + b_fc2


    # 出力層　　　　　　　　　
    W_fc3 = variable([4096, 10],"W_fc3")
    b_fc3 = variable([10],"b_fc3")
    y_conv = tf.matmul(X_fc2, W_fc3) + b_fc3
    #y_conv=tf.Print(y_conv,[y_conv])
    net=tf.nn.softmax(y_conv)


    return net



def Res_flow(inpt,t_now,delta_t,delta_w,task_name_tr):

    f_x = Res_func(inpt,task_name_tr,t_now)
    p_t = p(t_now)

    if task_name_tr == "Milstein_scheme":
        return inpt+p_t*delta_t*f_x +tf.pow(p_t*(1-p_t),0.5)*delta_w*f_x+mil()*(np.pow(delta_w,2)-delta_t)#ミルシュタインスキーム特有のやつ
    elif task_name_tr =="ODEnet" or task_name_tr=="test":
        return inpt+delta_t*f_x
    else:
        return inpt+p_t*delta_t*f_x +tf.pow(p_t*(1-p_t),0.5)*delta_w*f_x


def batch_norm(X, axes, shape, is_training):
    """
    バッチ正規化
    平均と分散による各レイヤの入力を正規化(白色化)する
    """
    if is_training is False:
        return X
    epsilon = 1e-8
    # 平均と分散
    mean, variance = tf.nn.moments(X, axes)
    # scaleとoffsetも学習対象
    scale = tf.Variable(tf.ones([shape]))
    offset = tf.Variable(tf.zeros([shape]))
    return tf.nn.batch_normalization(X, mean, variance, offset, scale, epsilon)


def hypernet(t,W1,W2,b1,b2):
    t=[[t]]
    W_h1=variable([1,100],"W_h1",True)
    b_h1=variable([100],"b_h1",True)
    x_h1=tf.nn.relu(tf.matmul(t, W_h1) + b_h1)
    W_h2=variable([100,100],"W_h2",True)
    b_h2=variable([100],"b_h2",True)
    x_h2=tf.nn.relu(tf.matmul(x_h1, W_h2) + b_h2)
    W_h3=variable([100,128],"W_h3",True)
    b_h3=variable([128],"b_h3",True)
    out = tf.matmul(x_h2, W_h3) + b_h3
    param= tf.nn.sigmoid(out)
    #ここから分割
    sigma1=param[0,0:64]
    sigma2=param[0,64:128]

    W1 = W1*sigma1
    b1 = b1*sigma1
    W2 = W2*sigma2
    b2 = b2*sigma2
    return W1,W2,b1,b2


def Res_func(inpt,task_name,t_now):
    global Z_imagetest
    if task_name=="test":
        is_training = False
    else:
        is_training = True
    W_conv1 = variable([3, 3, 64, 64],"W_conv1",True)
    b_conv1 = variable([64],"b_conv1",True)
    W_conv2 = variable([3, 3, 64, 64],"W_conv2",True)
    b_conv2 = variable([64],"b_conv2",True)
    # W_conv1,W_conv2,b_conv1,b_conv2=hypernet(t_now,W_conv1,W_conv2,b_conv1,b_conv2)
    inpt = batch_norm(inpt,[0,1,2],64,is_training)
    inpt_ = tf.nn.relu(conv2d(inpt, W_conv1)+b_conv1)
    inpt_ = batch_norm(inpt_,[0,1,2],64,is_training)


    output = conv2d(inpt_, W_conv2)+b_conv2



    return output
