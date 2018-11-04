import load_data
import tensorflow as tf
import numpy as np
import ResFlow as RF
import argparse
import os
import datetime



BATCH_SIZE = 64
LEARNING_RATE = 1e-6
EPOCH = 10
DATASET_DIRECTORY = "datasets"
MODEL_DIRECTORY = "model"
GPU = 0
#task_name="Fukasawa_scheme"
# task_name = "Simplicity_scheme"
# task_name = "Euler_Maruyama_scheme"
# task_name = "Milstein_scheme"
task_name = "ODEnet"

depth=52

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
    #X_train, Y_train = X_train[0:5000], Y_train[0:5000]
    #X_test, Y_test = X_test[0:1000] , Y_test[0:1000]

    X = tf.placeholder("float", [None, 32, 32, 3])
    Y = tf.placeholder("float", [None, 10])
    time_list = tf.placeholder("float", [None])
    W_list = tf.placeholder("float", [None])
    learning_rate = tf.placeholder("float", [])
    task_name_tr = tf.placeholder("string")

    net = RF.SDE_model(X,time_list,W_list,task_name)
    cross_entropy = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(net,1e-10,1.0)))
    #opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    opt=tf.train.GradientDescentOptimizer(learning_rate)
    train_op = opt.minimize(cross_entropy)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()
    batch_size = args.batch_size
    num_data = X_train.shape[0]
    
    print( "initial : %.3f " % sess.run(accuracy,feed_dict={
        X: X_test,
        Y: Y_test,
        time_list:t_test,
        W_list:W_test,
        task_name_tr:"test"}) )
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
                task_name_tr:task_name}
            #print(sess.run(net,feed_dict=feed_dict_train))
            #print(sess.run(tf.argmax(net, 1),feed_dict=feed_dict_train))
            sess.run([train_op], feed_dict=feed_dict_train)
            count = 0
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
            feed_dict_test={
                X: X_test,
                Y: Y_test,
                time_list:t_test,
                W_list:W_test,
                task_name_tr:"test"}
            print("saving checkpoint...")
            saver.save(sess,"model/model.ckpt"+"step"+str(j)+datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
            print("saved!")
            print( "accuracy after epoch %d : %.3f " % sess.run(accuracy,feed_dict=feed_dict_test))
           # accuracy_summary = tf.scalar_summary("accuracy", accuracy)
    

    sess.close()
import time
if __name__ == '__main__':
    start_time = time.time()
    run()
    elapsed  = time.time() - start_time
