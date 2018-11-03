# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""
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

def hypernet(t,W1,W2,b1,b2):
    t=[[float(t)]]
    W_h1=variable([1,100],"W_h1")
    b_h1=variable([100],"b_h1")
    x_h1=tf.nn.relu(tf.matmul(t, W_h1) + b_h1)
    W_h2=variable([100,100],"W_h2")
    b_h2=variable([100],"b_h2")
    x_h2=tf.nn.relu(tf.matmul(x_h1, W_h2) + b_h2)
    W_h3=variable([100,128],"W_h3")
    b_h3=variable([128],"b_h3")
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



    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def batch_norm(X, axes, shape, is_training):
    
    if is_training is False:
        return X
    epsilon = 1e-8
    # 平均と分散
    mean, variance = tf.nn.moments(X, axes)
    # scaleとoffsetも学習対象
    scale = tf.Variable(tf.ones([shape]))
    offset = tf.Variable(tf.zeros([shape]))
    return tf.nn.batch_normalization(X, mean, variance, offset, scale, epsilon)

def model(X,task):
    if task =="test":
        is_training= False
    else:
        is_training = True
    W_conv = variable([5, 5, 3, 64],"W_conv")
    b_conv = variable([64],"b_conv")
    X_image = tf.reshape(X, [-1,32,32,3])
    X_image = conv2d(X_image, W_conv)
    
    
    for i in range(15):
        W_conv1 = variable([5, 5, 64, 64],"W_conv1")
        b_conv1 = variable([64],"b_conv1")
        W_conv2 = variable([5, 5, 64, 64],"W_conv2")
        b_conv2 = variable([64],"b_conv2")
        
        
        
        W1,W2,b1,b2 = hypernet(i,W_conv1,W_conv2,b_conv1,b_conv2)
        X_image = batch_norm(X_image,[0,1,2],64,is_training)
        X_image = tf.nn.relu(conv2d(X_image, W1)+b1)
        X_image= batch_norm(X_image,[0,1,2],64,is_training)
        X_image = conv2d(X_image, W2)+b2 
        
    
    
    
    
    # 最大値プーリング(平均値のほうがよくない？)
    X_pool = tf.nn.max_pool(X_image, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding = "VALID")
    
    # 全結合層
    W_fc1 = variable([16* 16 * 64,4096],"W_fc1")# ここの7はちゃんとプーリング後の大きさを正しく計算する。
    b_fc1 = variable([4096],"b_fc1")
    X_pool_flat = tf.reshape(X_pool, [-1,  16* 16 * 64])#同じく
    X_fc1 = tf.nn.relu(tf.matmul(X_pool_flat, W_fc1) + b_fc1)
    
    # 出力層　　　　　　　　　
    W_fc2 = variable([4096, 10],"W_fc2")
    b_fc2 = variable([10],"b_fc2")
    y_conv = tf.matmul(X_fc1, W_fc2) + b_fc2
    y_conv=tf.Print(y_conv,[y_conv])
    net=tf.nn.softmax(y_conv)
    
    return net
    



X_train, Y_train, X_test, Y_test = load_data.load()    
X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])
task= tf.placeholder("string")
net = model(X,task)
cross_entropy = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(net,1e-10,1.0)))
#opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
opt=tf.train.GradientDescentOptimizer(1e-3)
train_op = opt.minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()
batch_size = 64
for j in range(10):
    feed_dict_train = {X:X_train[0:100],
                       Y:Y_train[0:100],
                       task:"train"}
    sess.run([train_op],feed_dict=feed_dict_train)
    print(sess.run(net,feed_dict=feed_dict_train))
    print(sess.run(tf.argmax(net, 1),feed_dict=feed_dict_train))
    print(sess.run(accuracy,feed_dict=feed_dict_train))
feed_dict_test = {X:X_test[0:100],
                  Y:Y_test[0:100],
                  task:"test"}
print(sess.run(accuracy,feed_dict=feed_dict_test))








