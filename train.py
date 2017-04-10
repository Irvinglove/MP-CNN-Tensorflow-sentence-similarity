#coding=utf-8
import tensorflow as tf
import numpy as np
from data_helper import read_data_sets,cul_feah_sim,cul_feaa_sim,dataset
# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("sick_dir", "SICK_data/SICK.txt", "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("sentence_length", 30, "Sentence length (default: 30)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 50)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# 得到权重,偏置,卷积,池化函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name='weights')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name='biases')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1],padding='VALID',name='conv')

def max_pool_22(x, widow_size):
    return tf.nn.max_pool(x, ksize=[1, 30 - widow_size + 1, 1, 1], strides=[1,30 - widow_size + 1,1,1], padding='VALID',name='pool')

def avg_pool_22(x, widow_size):
    return tf.nn.avg_pool(x, ksize=[1, 30 - widow_size + 1, 1, 1], strides=[1,30 - widow_size + 1,1,1], padding='VALID',name='pool')

def min_pool_22(x, widow_size):
    return -tf.nn.max_pool(-x, ksize=[1, 30 - widow_size + 1, 1, 1], strides=[1,30 - widow_size + 1,1,1], padding='VALID',name='pool')

# 得到数据
s1_train ,s2_train ,label_train, s1_test, s2_test, label_test, embedding_w = \
    read_data_sets('MNIST_data', FLAGS)

x1 = tf.placeholder(tf.int64, shape=[None, 30], name='x1_input')
x2 = tf.placeholder(tf.int64, shape=[None, 30], name='x2_input')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_input')

s1_image = tf.nn.embedding_lookup(embedding_w, x1)
s2_image = tf.nn.embedding_lookup(embedding_w, x2)
x1_flat = tf.reshape(s1_image, [-1, 30, 50, 1])
x2_flat = tf.reshape(s1_image, [-1, 30, 50, 1])

# windows_size = 30
Ws130 = weight_variable([30,50,1,100])
bs130 = bias_variable([100])
hs130_x1 = tf.nn.relu(conv2d(x1_flat, Ws130) + bs130)
hs130_x2 = tf.nn.relu(conv2d(x2_flat, Ws130) + bs130)
maxpool_s130_x1 = max_pool_22(hs130_x1, 30)
maxpool_s130_x2 = max_pool_22(hs130_x2, 30)
avgpool_s130_x1 = avg_pool_22(hs130_x1, 30)
avgpool_s130_x2 = avg_pool_22(hs130_x2, 30)
minpool_s130_x1 = min_pool_22(hs130_x1, 30)
minpool_s130_x2 = min_pool_22(hs130_x2, 30)


# sentence 1 , windows_size = 1
Ws11 = weight_variable([1,50,1,100])
bs11 = bias_variable([100])
hs11_x1 = tf.nn.relu(conv2d(x1_flat, Ws11) + bs11)
hs11_x2 = tf.nn.relu(conv2d(x2_flat, Ws11) + bs11)
maxpool_s11_x1 = max_pool_22(hs11_x1, 1)
maxpool_s11_x2 = max_pool_22(hs11_x2, 1)
avgpool_s11_x1 = avg_pool_22(hs11_x1, 1)
avgpool_s11_x2 = avg_pool_22(hs11_x2, 1)
minpool_s11_x1 = min_pool_22(hs11_x1, 1)
minpool_s11_x2 = min_pool_22(hs11_x2, 1)


# sentence 1 , windows_size = 2
Ws12 = weight_variable([2,50,1,100])
bs12 = bias_variable([100])
hs12_x1 = tf.nn.relu(conv2d(x1_flat, Ws12) + bs12)
hs12_x2 = tf.nn.relu(conv2d(x2_flat, Ws12) + bs12)
maxpool_s12_x1 = max_pool_22(hs12_x1, 2)
maxpool_s12_x2 = max_pool_22(hs12_x2, 2)
avgpool_s12_x1 = avg_pool_22(hs12_x1, 2)
avgpool_s12_x2 = avg_pool_22(hs12_x2, 2)
minpool_s12_x1 = min_pool_22(hs12_x1, 2)
minpool_s12_x2 = min_pool_22(hs12_x2, 2)

maxsim_feah = cul_feah_sim(maxpool_s11_x1, maxpool_s11_x2, maxpool_s12_x1, maxpool_s12_x2, maxpool_s130_x1, maxpool_s130_x2)
avgsim_feah = cul_feah_sim(avgpool_s11_x1, avgpool_s11_x2, avgpool_s12_x1, avgpool_s12_x2, avgpool_s130_x1, avgpool_s130_x2)
minsim_feah = cul_feah_sim(minpool_s11_x1, minpool_s11_x2, minpool_s12_x1, minpool_s12_x2, minpool_s130_x1, minpool_s130_x2)

maxsim_feaa = cul_feaa_sim(maxpool_s11_x1, maxpool_s11_x2, maxpool_s12_x1, maxpool_s12_x2, maxpool_s130_x1, maxpool_s130_x2)
avgsim_feaa = cul_feaa_sim(avgpool_s11_x1, avgpool_s11_x2, avgpool_s12_x1, avgpool_s12_x2, avgpool_s130_x1, avgpool_s130_x2)
minsim_feaa = cul_feaa_sim(minpool_s11_x1, minpool_s11_x2, minpool_s12_x1, minpool_s12_x2, minpool_s130_x1, minpool_s130_x2)

fea = tf.concat([maxsim_feah,avgsim_feah,minsim_feah, maxsim_feaa,avgsim_feaa,minsim_feaa],3)
fea_flat = tf.reshape(fea,[-1,327])


W_fc1 = weight_variable([327, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(fea_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

sess = tf.InteractiveSession()
loss = tf.reduce_sum(tf.square(y_conv - y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
_, pearson = tf.contrib.metrics.streaming_pearson_correlation(y_conv, y_)
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

STS_train = dataset(s1=s1_train, s2=s2_train, label=label_train)
print "初始化完毕，开始训练"
for i in range(20000):
    batch_train = STS_train.next_batch(50)
    # 训练模型
    train_step.run(feed_dict={x1: batch_train[0], x2:batch_train[1], y_: batch_train[2], keep_prob: 0.5})
    # 对结果进行记录
    if i % 100 == 0:
        train_loss = loss.eval(feed_dict={
            x1: batch_train[0], x2: batch_train[1], y_: batch_train[2], keep_prob: 1.0})
        train_pearson = pearson.eval(feed_dict={
                    x1: batch_train[0], x2: batch_train[1], y_: batch_train[2], keep_prob: 1.0})
        print "step %d, training pearson %g, loss %g" % (i, train_pearson, train_loss)

# STS_test = data_helper.dataset(s1=s1_test, s2=s2_test, label=label_test)
print "test pearson %g"%pearson.eval(feed_dict={
    x1: s1_test, x2: s2_test, y_: label_test, keep_prob: 1.0})
print "test loss %g"%loss.eval(feed_dict={
    x1: s1_test, x2: s2_test, y_: label_test, keep_prob: 1.0})


















