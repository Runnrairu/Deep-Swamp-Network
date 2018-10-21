import load_data
import tensorflow as tf
import numpy as np
import ResFlow as RF
import argparse
import os
BATCH_SIZE = 20
LEARNING_RATE = 1e-4
DATASET_DIRECTORY = "datasets"
MODEL_DIRECTORY = "model"
GPU = 0
task_name="Fukasawa_scheme"
# task_name = "Simplicity_scheme"
# task_name = "Euler_Maruyama_scheme"
# task_name = "Milstein_scheme"
# task_name = "ODEnet"

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
    
    X = tf.placeholder("float", [None, 32, 32, 3])
    Y = tf.placeholder("float", [None, 10])
    time_list = tf.placeholder("float", [None])
    W_list = tf.placeholder("float", [None])
    learning_rate = tf.placeholder("float", [])
    task_name_tr = tf.placeholder("string")
    
    net = RF.SDE_model(X,time_list,W_list,task_name)
    cross_entropy = -tf.reduce_sum(Y*tf.log(net))
    #opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    opt=tf.train.GradientDescentOptimizer(learning_rate)
    train_op = opt.minimize(cross_entropy)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()
    batch_size = args.batch_size
    
    for j in range (1):
        
        for i in range (0, 500, batch_size):
            print(i)
            t,W = RF.tW_def(depth,task_name)
            feed_dict_train={
                X: X_train[i:i + batch_size], 
                Y: Y_train[i:i + batch_size],
                learning_rate: args.learning_rate,
                time_list:t,
                W_list:W,
                task_name_tr:task_name}
            sess.run([train_op], feed_dict=feed_dict_train)
            if i % 512 == 0:
                
                saver.save(sess, 'progress', global_step=i)
    print("test")
    for i in range (0, 10000, batch_size):
        if i + batch_size < 10000:
            t,W = RF.tW_def(depth,task_name)
            acc = sess.run([accuracy],feed_dict={
                X: X_test[i:i+batch_size],
                Y: Y_test[i:i+batch_size],
                time_list:t,
                W_list:W,
                task_name:task_name,
                })
            accuracy_summary = tf.scalar_summary("accuracy", accuracy)
            print(acc)

    sess.close()
 
if __name__ == '__main__':
    run()
   
