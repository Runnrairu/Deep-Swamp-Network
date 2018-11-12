import load_data
import tensorflow as tf
import numpy as np
import ResFlow as RF
import argparse
import os
import datetime


HYPER_NET = "N"
#HYPER_NET = "1"
#HYPER_NET = "2"
BATCH_SIZE = 64
LEARNING_RATE = 1e-6
EPOCH = 10
DATASET_DIRECTORY = "datasets"
MODEL_DIRECTORY = "model"
SAVE_ENABLE = False
GPU = 0
#task_name="Fukasawa_scheme"
# task_name = "Simplicity_scheme"
# task_name = "Euler_Maruyama_scheme"
# task_name = "Milstein_scheme"
# task_name = "ResNet"
# task_name = "Stochastic_Depth"
# task_name = "ResNet_test"
# task_name = "test"
task_name = "ODEnet"

depth=52

hypernet_variable = (["W_conv1","b_conv1","W_conv2","b_conv2"],
            ["W_conv1","b_conv1","W_conv2","b_conv2","W_h1","b_h1","W_h2","b_h2"],
            ["W_h1_2","b_h1_2","W_h2_2","b_h2_2"])



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--model_directory', type=str, default=MODEL_DIRECTORY)
    parser.add_argument('-dd', '--dataset_directory', type=str, default=DATASET_DIRECTORY)
    parser.add_argument('-bs', '--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('-g', '--gpu', type=int, default=GPU)

    args = parser.parse_args()
    directory_output = os.path.join(args.model_directory)

    X_train, Y_train, X_test, Y_test = load_data.load()

    #縮小する
    X_train, Y_train = X_train[0:5000], Y_train[0:5000]
    X_test, Y_test = X_test[0:1000] , Y_test[0:1000]

    X = tf.placeholder("float", [None, 32, 32, 3])
    Y = tf.placeholder("float", [None, 10])
    time_list = tf.placeholder("float", [None])
    W_list = tf.placeholder("float", [None])
    learning_rate = tf.placeholder("float", [])
    task_name_tr = tf.placeholder("string")
    hypernet=HYPER_NET
    net = RF.SDE_model(X,time_list,W_list,task_name,hypernet)
    cross_entropy = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(net,1e-10,1.0)))
    #opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    var_name_list1 = ["W_conv","b_conv"]+hypernet_variable[0]
    var_name_list2 = ["W_fc1","b_fc1","W_fc2","b_fc2","W_fc3","b_fc3"]

    train_op = None

    sess = tf.Session()

    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()
    batch_size = args.batch_size
    num_data = X_train.shape[0]


    with tf.variable_scope("scope", reuse=True ):
        var_list1 = [ tf.get_variable(name=x) for x in var_name_list1 ]
        var_list2 = [ tf.get_variable(name=x) for x in var_name_list2 ]

    if task_name == "ResNet" or task_name =="ResNet_test" or task_name =="Stochastic_Depth":
        learning_late = 1e-4
    else:
        learning_late = 1e-6
#    train_op1 = tf.train.MomentumOptimizer( 1e-6 , 0.9 ).minimize(cross_entropy,var_list = var_list1 )  # tf.train.GradientDescentOptimizer(0.000001)
#    train_op2 = tf.train.MomentumOptimizer( 1e-6 , 0.9 ).minimize(cross_entropy,var_list = var_list2 ) # tf.train.GradientDescentOptimizer(0.0001)
    train_op = tf.train.MomentumOptimizer( learning_late , 0.9 ).minimize(cross_entropy) # tf.group(train_op1, train_op2)  # tf.train.GradientDescentOptimizer( 1e-6 ).minimize(cross_entropy) #

    sess.run(tf.global_variables_initializer())

    print(tf.trainable_variables())

    for j in range (EPOCH):
        sff_idx = np.random.permutation(num_data)
        for idx in range(0, num_data, batch_size):
            batch_x = X_train[sff_idx[idx: idx + batch_size
            if idx + batch_size < num_data else num_data]]
            batch_y = Y_train[sff_idx[idx: idx + batch_size
            if idx + batch_size < num_data else num_data]]

            t,W = RF.tW_def(depth,task_name)

            feed_dict_train={
                X: batch_x,
                Y: batch_y,
                learning_rate: args.learning_rate,
                time_list:t,
                W_list:W,
                task_name_tr:task_name
                }

            #print(sess.run(net,feed_dict=feed_dict_train))
            #print(sess.run(tf.argmax(net, 1),feed_dict=feed_dict_train))

            sess.run([train_op], feed_dict=feed_dict_train)
            
            #for z in (RF.Z_imagetest):
            #print(sess.run(net,feed_dict= feed_dict_train))
                #assert(not np.isnan(sess.run(z,feed_dict=feed_dict_train)).any())
                #count += 1
        elapsed  = time.time() - start_time
        print("epoch %d end : %.3f seconds elapsed " % (j,elapsed) )

            #if j % 512 == 0:
            #    a=1
        if j == 0 or j % 10 == 9 or j+1==EPOCH : # 最初 , 10回ごと , 最後　のどれかならテストしてみる
            t_test,W_test = RF.tW_def(depth,"test")
            if task_name == "ResNet" or task_name =="Stochastic_Depth":
                task_name_test = "ResNet_test"
            else:
                task_name_test = "test"
            feed_dict_test={
                X: X_test,
                Y: Y_test,
                time_list:t_test,
                W_list:W_test,
                task_name_tr:task_name_test
                }
            if SAVE_ENABLE :
                print("saving checkpoint...")
                saver.save(sess,"model/model.ckpt"+"step"+str(j)+datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
                print("saved!")
            print( "accuracy after epoch %d : %.3f " % (j,sess.run(accuracy,feed_dict=feed_dict_test) ))
           # accuracy_summary = tf.scalar_summary("accuracy", accuracy)
    #ここからパラメータ数計算および列挙
    total_parameters = 0
    parameters_string = ""
    for variable in tf.trainable_variables():

        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d, " % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

    print(parameters_string)
    print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
    sess.close()
import time
if __name__ == '__main__':
    start_time = time.time()
    run()
    elapsed  = time.time() - start_time
