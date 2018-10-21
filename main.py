import load_data
import tensorflow as tf
import numpy as np
import ResFlow as RF

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
DATASET_DIRECTORY = "datasets"
MODEL_DIRECTORY = "model"
GPU = 0
task_name="Fukasawa_scheme"
# task_name = "Simplicity_scheme"
# task_name = "Euler_Maruyama_scheme"
# task_name = "Milstein_scheme"
# task_name = "ODEnet"


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--model_directory', type=str, default=MODEL_DIRECTORY)
    parser.add_argument('-dd', '--dataset_directory', type=str, default=DATASET_DIRECTORY)
    parser.add_argument('-bs', '--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('-g', '--gpu', type=int, default=GPU)
    args = parser.parse_args()
    directory_output = os.path.join(args.model_directory, args.experiment_id)
    
    X_train, Y_train, X_test, Y_test = load_data.load()
    
    X = tf.placeholder("float", [batch_size, 32, 32, 3])
    Y = tf.placeholder("float", [batch_size, 10])
    t = tf.placeholder("float", [None])
    W = tf.placeholder("float", [None])
    learning_rate = tf.placeholder("float", [])
    task_name = tf.placeholder("string")
    
    net = RF.SDE_model(X,t,W,task_name)
    cross_entropy = -tf.reduce_sum(Y*tf.log(net))
    opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = opt.minimize(cross_entropy)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()
    batch_size = args.batch_size
    
    for j in range (10):
        for i in range (0, 50000, batch_size):
            t,W = RF.tW_def(10,task_name)
            feed_dict={
                X: X_train[i:i + batch_size], 
                Y: Y_train[i:i + batch_size],
                learning_rate: args.learning_rate,
                time_list:t,
                W_list:W,
                task_name:task_name}
            sess.run([train_op], feed_dict=feed_dict)
            if i % 512 == 0:
                print "training on image #%d" % i
                saver.save(sess, 'progress', global_step=i)

    for i in range (0, 10000, batch_size):
        if i + batch_size < 10000:
            t,W = RF.tW_def(10,task_name)
            acc = sess.run([accuracy],feed_dict={
                X: X_test[i:i+batch_size],
                Y: Y_test[i:i+batch_size],
                time_list:t,
                W_list:W,
                task_name:task_name,
                })
            accuracy_summary = tf.scalar_summary("accuracy", accuracy)
            print acc

    sess.close()
 
if __name__ == '__main__':
    run()
   
