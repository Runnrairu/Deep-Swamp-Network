# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

T = 30.0

d=1

VARIANCE = 1e-5


def p(t):
    p_T = 0.3

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
    elif task_name == "Stochastic_Depth":
        t,W = S_depth(n,T)
    elif task_name == "Euler_Maruyama_scheme" or task_name == "Milstein_scheme":
        t,W = Euler_Maruyama_scheme(n,T)
    elif task_name == "ODEnet" or task_name=="test" or task_name =="ResNet" or task_name =="ResNet_test":
        t,W = ODEnet(n,T)

    else:
        print("Invarid!")
    return t,W



def tW_def_test(n,task_name):
    t=[]
    W=[]
    if task_name=="test" or task_name =="ResNet_test":
        t,W = ODEnet(n,T)
    elif task_name== "Stochastic_Depth_test":
        t,W = StochasticDepth_test(n,T)
    return t,W


def StochasticDepth_test(n,T):
    t_now = 0
    delta_t = float(T)/(n+1)
    t = [delta_t]*(n+1)
    W=[0]*(n+1)    
    for i in range(n+1):
        W[i]=p(t_now)
        t_now +=delta_t
    
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


def S_depth(n,T):
    delta_t = float(T)/(n+1)
    t = [delta_t]*(n+1)
    W=[0.5]*(n+1)
    a=[1,0]
    t_now = 0
    for i in range(n+1):
        p_t=p(t_now)
        W[i]=np.random.choice(a, size=None, replace=True, p=[p_t,1-p_t])
        t_now += delta_t
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



def variable(shape,var_name,Flow=False,init=None):

    v=np.sqrt(VARIANCE)
    long = len(shape)
    

    with tf.variable_scope('scope',reuse=tf.AUTO_REUSE):
        if init is None:
            if long == 1:
                initial = tf.constant(0.0,shape=shape,dtype="float32")
                var = tf.get_variable(name=var_name,initializer = initial)
            else:
                initial = tf.truncated_normal_initializer(stddev=v)
                var = tf.get_variable(name=var_name,shape=shape,initializer = initial)
        else:
            if isinstance(init,tf.Tensor):
                var = tf.get_variable(name=var_name,initializer = init )
            else:
                var = tf.get_variable(name=var_name,shape=shape,initializer = init )
    return var



def SDE_model(X,depth,t,W,task_name_tr,hypernet,test=False):


    
    
    X_image = tf.reshape(X, [-1,32,32,3])

    X_image=tf.tile(X_image,[1,1,1,170])
    
    t_now = 0

    for i in range(depth):
        delta_t = t[i]
        delta_W = W[i]
        t_now += delta_t
        X_image = Res_flow(X_image,t_now,delta_t,delta_W,task_name_tr,i,hypernet,test)
        #X_image=tf.Print(X_image,[X_image])




    
    X_pool = tf.nn.avg_pool(X_image, ksize=[1, 4, 4, 1],strides=[1, 4, 4, 1],padding = "VALID")
    
    # 全結合層
    W_fc1 = variable([8* 8 * 510,1024],"W_fc1")# ここの7はちゃんとプーリング後の大きさを正しく計算する。
    b_fc1 = variable([1024],"b_fc1")
    X_pool_flat = tf.reshape(X_pool, [-1,  8* 8 * 510])#同じく
    X_fc1 = tf.nn.swish(tf.matmul(X_pool_flat, W_fc1) + b_fc1)

    #W_fc2 = variable([1024,1024],"W_fc2")
    #b_fc2 = variable([1024],"b_fc2")
    #X_fc2 = tf.matmul(X_fc1, W_fc2) + b_fc2


    # 出力層　　　　　　　　　
    W_fc3 = variable([1024, 10],"W_fc3")
    b_fc3 = variable([10],"b_fc3")
    y_conv = tf.matmul(X_fc1, W_fc3) + b_fc3
    #y_conv=tf.Print(y_conv,[y_conv])
    net=tf.nn.softmax(y_conv)


    return net



def Res_flow(inpt,t_now,delta_t,delta_w,task_name_tr,count,hypernet,f_test):
    if task_name_tr=="Stochastic Depth" and delta_w==0:
        f_x=0
    elif task_name_tr == "Milstein_scheme":
        f_x,wxb=Res_func(inpt,task_name_tr,t_now,count,hypernet,f_test)
        
        
    else:
        f_x,wxb = Res_func(inpt,task_name_tr,t_now,count,hypernet,f_test)
     
    p_t = p(t_now)

    if task_name_tr == "Milstein_scheme":
        
        return inpt+p_t*delta_t*f_x +tf.pow(p_t*(1-p_t),0.5)*delta_w*f_x+mil_fx(inpt,task_name_tr,t_now,count,hypernet,f_test,f_x,wxb)*(np.pow(delta_w,2)-delta_t)#ミルシュタインスキーム特有のやつ
    elif task_name_tr =="ODEnet" or task_name_tr=="test" or task_name_tr =="ResNet" or task_name_tr =="ResNet_test":
        return inpt+delta_t*f_x
    elif task_name_tr=="Stochastic Depth":
        return inpt+delta_t*delta_w*f_x
    else:
        return inpt+p_t*delta_t*f_x +tf.pow(p_t*(1-p_t),0.5)*delta_w*f_x


def batch_norm(X, axes, shape, is_training , id ,hypernet):
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


    scale = variable([shape],init=tf.ones([shape]),var_name="batch_scale_%s" % id )
    offset = variable([shape],init=tf.zeros([shape]),var_name="batch_offset_%s" % id )
    return tf.nn.batch_normalization(X, mean, variance, offset, scale, epsilon)

def hypernet2(t):
    t=[[t]]
    conv=3*3*510*510
    bias=510
    W_h1=variable([1,10],"W_h1",True)
    b_h1=variable([10],"b_h1",True)
    x_h1=tf.nn.swish(tf.matmul(t, W_h1) + b_h1)
    W_h3=variable([10,2*(conv+bias)],"W_h3",True)
    b_h3=variable([2*(conv+bias)],"b_h3",True)
    out = tf.matmul(x_h1, W_h3) + b_h3
    #param= tf.nn.sigmoid(out)

    #ここから分割
    W1=out[0,0:conv]
    b1=out[0,conv:conv+bias]
    W2=out[0,conv+bias:2*conv+bias]
    b2=out[0,2*conv+bias:2*(conv+bias)]
    W1=tf.reshape(W1,[3,3,510,510])
    W2=tf.reshape(W2,[3,3,510,510])

    return W1,W2,b1,b2

def hypernet1(t,W1,W2,b1,b2):
    t=[[t]]
    W_h1=variable([1,100],"W_h1",True)
    b_h1=variable([100],"b_h1",True)
    x_h1=tf.nn.swish(tf.matmul(t, W_h1) + b_h1)
    W_h2=variable([100,100],"W_h2",True)
    b_h2=variable([100],"b_h2",True)
    x_h2=tf.nn.swish(tf.matmul(x_h1, W_h2) + b_h2)
    W_h3=variable([100,1020],"W_h3",True)
    b_h3=variable([1020],"b_h3",True)
    out = tf.matmul(x_h2, W_h3) + b_h3
    param= tf.nn.sigmoid(out)
    #ここから分割
    sigma1=param[0,0:510]
    sigma2=param[0,510:1020]

    W1 = W1*sigma1
    b1 = b1*sigma1
    W2 = W2*sigma2
    b2 = b2*sigma2
    return W1,W2,b1,b2


def mil_fx(inpt,task_name_tr,t_now,count,hypernet,f_test,f_x,wxb):
    is_training = not f_test
    
        
    if hypernet == "N" or hypernet == "1":
        W_conv1 = variable([3, 3, 66, 66],"W_conv1",True)
        b_conv1 = variable([66],"b_conv1",True)
        W_conv2 = variable([3, 3, 66, 66],"W_conv2",True)
        b_conv2 = variable([66],"b_conv2",True)
        if hypernet=="1":
            W_conv1,W_conv2,b_conv1,b_conv2=hypernet1(t_now,W_conv1,W_conv2,b_conv1,b_conv2)
    elif hypernet=="2":
        W_conv1,W_conv2,b_conv1,b_conv2=hypernet2(t_now)

    #inpt = batch_norm(inpt,[0,1,2],66,is_training,"1",hypernet=hypernet)
    inpt_ = conv2d(f_x, W_conv1)
    #inpt_ = batch_norm(inpt_,[0,1,2],66,is_training,"2",hypernet=hypernet)


    inpt_=tf.matmul(wxb,inpt_)
    output = conv2d(inpt_, W_conv2)



def Res_func(inpt,task_name,t_now,count,hypernet,f_test):
    is_training = not f_test
    wxb=0
    if task_name == "ResNet" or task_name=="Stochastic_Depth" or task_name=="ResNet_test" :
        W_conv1 = variable([3, 3, 510, 510],"W_conv1_"+str(count),True)
        b_conv1 = variable([510],"b_conv1_"+str(count),True)
        W_conv2 = variable([3, 3, 510, 510],"W_conv2_"+str(count),True)
        b_conv2 = variable([510],"b_conv2_"+str(count),True)

    elif hypernet == "N" or hypernet == "1":
        W_conv1 = variable([3, 3, 510, 510],"W_conv1",True)
        b_conv1 = variable([510],"b_conv1",True)
        W_conv2 = variable([3, 3, 510, 510],"W_conv2",True)
        b_conv2 = variable([510],"b_conv2",True)
        if hypernet=="1":
            W_conv1,W_conv2,b_conv1,b_conv2=hypernet1(t_now,W_conv1,W_conv2,b_conv1,b_conv2)
    elif hypernet=="2":
        W_conv1,W_conv2,b_conv1,b_conv2=hypernet2(t_now)

    if task_name == "ResNet" or task_name =="ResNet_test" or task_name =="Stochastic_Depth":
        #inpt = batch_norm(inpt,[0,1,2],66,is_training,"1_"+str(count),hypernet=hypernet)
        inpt_ = tf.nn.swish(conv2d(inpt, W_conv1)+b_conv1)
        #inpt_ = batch_norm(inpt_,[0,1,2],66,is_training,"2_"+str(count),hypernet=hypernet)
    elif task_name == "Milstein_scheme":
        
        wxb=conv2d(inpt, W_conv1)+b_conv1
        inpt_=tf.nn.swish(wxb)
    else:
        #inpt = batch_norm(inpt,[0,1,2],66,is_training,"1",hypernet=hypernet)
        inpt_ = tf.nn.swish(conv2d(inpt, W_conv1)+b_conv1)
        #inpt_ = batch_norm(inpt_,[0,1,2],66,is_training,"2",hypernet=hypernet)


    output = conv2d(inpt_, W_conv2)+b_conv2



    return output,wxb
