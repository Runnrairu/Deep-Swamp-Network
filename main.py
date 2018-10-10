import load_data
import tensorflow as tf
import numpy as np

BATCH_SIZE = 64
LEARNING_RATE = 1e-4

def run():
    
    
    X_train, Y_train, X_test, Y_test = load_data.load()
    
    X = tf.placeholder("float", [batch_size, 32, 32, 3])
    Y = tf.placeholder("float", [batch_size, 10])
    learning_rate = tf.placeholder("float", [])
    cross_entropy = -tf.reduce_sum(Y*tf.log(net))
    opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = opt.minimize(cross_entropy)
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()
    for j in range (10):
        for i in range (0, 50000, batch_size):
            feed_dict={
                X: X_train[i:i + batch_size], 
                Y: Y_train[i:i + batch_size],
                learning_rate: 0.001}
            sess.run([train_op], feed_dict=feed_dict)
            if i % 512 == 0:
                print "training on image #%d" % i
                saver.save(sess, 'progress', global_step=i)

    for i in range (0, 10000, batch_size):
        if i + batch_size < 10000:
            acc = sess.run([accuracy],feed_dict={
                X: X_test[i:i+batch_size],
                Y: Y_test[i:i+batch_size]
            })
            accuracy_summary = tf.scalar_summary("accuracy", accuracy)
            print acc

    sess.close()
    
    
