# 一般的なGPUを使うとき（メモリが潤沢でない場合）用のコード
import time
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
LEARNING_RATE = 1e-5
EPOCH = 100
DATASET_DIRECTORY = "datasets"
MODEL_DIRECTORY = "model"
SAVE_ENABLE = False
GPU = 0
# task_name="Fukasawa_scheme"
# task_name = "Simplicity_scheme"
# task_name = "Euler_Maruyama_scheme"
# task_name = "Milstein_scheme"
# task_name = "ResNet"
# task_name = "Stochastic_Depth"
# task_name = "ResNet_test"
# task_name = "test"
task_name = "ODENet"

depth = 52

hypernet_variable = (["W_conv1", "b_conv1", "W_conv2", "b_conv2"],
                     ["W_conv1", "b_conv1", "W_conv2", "b_conv2",
                         "W_h1", "b_h1", "W_h2", "b_h2"],
                     ["W_h1_2", "b_h1_2", "W_h2_2", "b_h2_2"])


def run():
    global task_name
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--model_directory',
                        type=str, default=MODEL_DIRECTORY)
    parser.add_argument('-dd', '--dataset_directory',
                        type=str, default=DATASET_DIRECTORY)
    parser.add_argument('-bs', '--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('-lr', '--learning_rate',
                        type=float, default=LEARNING_RATE)
    parser.add_argument('-g', '--gpu', type=int, default=GPU)
    parser.add_argument('-t', '--task_name', type=str, default=task_name)
    parser.add_argument('-n', '--hyper_net', type=str, default=HYPER_NET)
    parser.add_argument('-v', '--variance', type=float, default=RF.VARIANCE)
    parser.add_argument('-de', '--depth', type=int, default=52)
    parser.add_argument('-ep', '--epoch', type=int, default=100)

    args = parser.parse_args()
    epoch = args.epoch
    task_name = args.task_name
    RF.VARIANCE = args.variance
    directory_output = os.path.join(args.model_directory)
    depth = args.depth

    X_train, Y_train, X_test, Y_test = load_data.load()
    X_test_m = [0]*(10)
    Y_test_m = [0]*(10)
    for i in range(10):
        X_test_m[i] = X_test[i*1000:(i+1)*1000]
        Y_test_m[i] = Y_test[i*1000:(i+1)*1000]

    # 縮小する
    #X_train, Y_train = X_train[0:5000], Y_train[0:5000]
    #X_test, Y_test = X_test[0:1000] , Y_test[0:1000]

    X = tf.placeholder("float", [None, 32, 32, 3])
    Y = tf.placeholder("float", [None, 10])
    time_list = tf.placeholder("float", [None])
    W_list = tf.placeholder("float", [None])
    learning_rate = tf.placeholder("float", [])
    hypernet = args.hyper_net  # tf.placeholder("string")
    task_name_tr = tf.placeholder("string")

    net = RF.SDE_model(X, depth, time_list, W_list,
                       task_name, hypernet, test=False)
    test_net = RF.SDE_model(X, depth, time_list, W_list,
                            task_name, hypernet, test=True)

    sess = tf.Session()
    beta = 1e-3

    cross_entropy = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(net, 1e-10, 1.0)))
    suml2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    loss = cross_entropy + beta * suml2
    #opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    var_name_list1 = ["W_conv", "b_conv"]+hypernet_variable[0]
    var_name_list2 = ["W_fc1", "b_fc1", "W_fc2", "b_fc2", "W_fc3", "b_fc3"]

    #train_op = None

    correct_prediction = tf.equal(tf.argmax(test_net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()
    batch_size = args.batch_size
    num_data = X_train.shape[0]

    # with tf.variable_scope("scope", reuse=True ):
    #    var_list1 = [ tf.get_variable(name=x) for x in var_name_list1 ]
    #    var_list2 = [ tf.get_variable(name=x) for x in var_name_list2 ]

#    if task_name == "ResNet" or task_name =="ResNet_test" or task_name =="Stochastic_Depth":
#        learning_late = 1e-4
#    else:
#        learning_late = 1e-6
#    train_op1 = tf.train.MomentumOptimizer( 1e-6 , 0.9 ).minimize(cross_entropy,var_list = var_list1 )  # tf.train.GradientDescentOptimizer(0.000001)
#    train_op2 = tf.train.MomentumOptimizer( 1e-6 , 0.9 ).minimize(cross_entropy,var_list = var_list2 ) # tf.train.GradientDescentOptimizer(0.0001)
    # tf.group(train_op1, train_op2)  # tf.train.GradientDescentOptimizer( 1e-6 ).minimize(cross_entropy) #
    train_op = tf.train.MomentumOptimizer(
        args.learning_rate, 0.9).minimize(loss)

    sess.run(tf.global_variables_initializer())

    print(tf.trainable_variables())
    late_ad = 1.0
    for j in range(epoch):
        sff_idx = np.random.permutation(num_data)
        if j < 20:
            late_ad = 1.0
        elif j < 40:
            late_ad = 0.1
        elif j < 60:
            late_ad = 0.01
        else:
            late_ad = 0.001

        for idx in range(0, num_data, batch_size):
            batch_x = X_train[sff_idx[idx: idx + batch_size
                                      if idx + batch_size < num_data else num_data]]
            batch_y = Y_train[sff_idx[idx: idx + batch_size
                                      if idx + batch_size < num_data else num_data]]

            t, W = RF.tW_def(depth, task_name)

            feed_dict_train = {
                X: batch_x,
                Y: batch_y,
                learning_rate: args.learning_rate*late_ad,
                time_list: t,
                W_list: W,
                task_name_tr: task_name}

            # print(sess.run(net,feed_dict=feed_dict_train))
            #print(sess.run(tf.argmax(net, 1),feed_dict=feed_dict_train))

            sess.run([train_op], feed_dict=feed_dict_train)
            # for z in (RF.Z_imagetest):
            #print(sess.run(net,feed_dict= feed_dict_train))
            #assert(not np.isnan(sess.run(z,feed_dict=feed_dict_train)).any())
            #count += 1
        elapsed = time.time() - start_time
        print("epoch %d end : %.3f seconds elapsed " % (j, elapsed))

        # if j % 512 == 0:
        #    a=1
        if True or j == 0 or j % 10 == 9 or j+1 == EPOCH:  # 最初 , 10回ごと , 最後　のどれかならテストしてみる
            t_test, W_test = RF.tW_def(depth, "test")
            if task_name == "ResNet" or task_name == "Stochastic_Depth":
                task_name_test = "ResNet_test"
            else:
                task_name_test = "test"
            feed_dict_test = {
                X: X_test,
                Y: Y_test,
                time_list: t_test,
                W_list: W_test,
                task_name_tr: task_name_test
            }
            if SAVE_ENABLE:
                print("saving checkpoint...")
                saver.save(sess, "model/model.ckpt"+str(task_name)+"step" +
                           str(j)+datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
                print("saved!")
            acc = 0
            for i in range(10):
                feed_dict_test = {
                    X: X_test_m[i],
                    Y: Y_test_m[i],
                    time_list: t_test,
                    W_list: W_test,
                    task_name_tr: task_name_test
                }
                acc += sess.run(accuracy, feed_dict=feed_dict_test)
            acc = acc/10.0
            print("accuracy after epoch %d : %.3f " % (j, acc), flush=True)
           # accuracy_summary = tf.scalar_summary("accuracy", accuracy)
    # ここからパラメータ数計算および列挙
    total_parameters = 0
    parameters_string = ""
    for variable in tf.trainable_variables():
        sess.run(tf.verify_tensor_all_finite(
            variable,
            "NaN  in : %s \n" % variable.name
        ))
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d, " %
                                  (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d, " %
                                  (variable.name, str(shape), variable_parameters))

    print(parameters_string)
    print("Total %d variables, %s params" %
          (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
    sess.close()


if __name__ == '__main__':
    start_time = time.time()
    run()
    elapsed = time.time() - start_time
