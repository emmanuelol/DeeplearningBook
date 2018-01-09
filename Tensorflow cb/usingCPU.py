import tensorflow as tf

config=tf.ConfigProto(log_device_placement=True)

with tf.device('/cpu:0'):
    rand_t=tf.random_uniform([50,50],0,10,dtype=tf.float32,seed=0)
    a=tf.Variable(rand_t)
    b=tf.Variable(rand_t)
    c=tf.matmul(a,b)
    init=tf.global_variables_initializer()

sess=tf.Session(config)
sess.run(init)
print(sess.run(c))
sess.close()
