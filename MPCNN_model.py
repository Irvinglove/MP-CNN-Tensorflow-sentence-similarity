import tensorflow as tf
import numpy as np
from data_helper import build_glove_dic

class TextMPCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, filter_size, h_pool_type, p_pool_type,
                 h_ws_sizes, p_ws_sizes, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_s1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_s1")
        self.input_s2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_s2")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.filter_size = filter_size
        self.h_pool_type = h_pool_type
        self.p_pool_type = p_pool_type
        self.h_ws_sizes = h_ws_sizes
        self.p_ws_sizes = p_ws_sizes
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.l2_reg_lambda = l2_reg_lambda
        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)

        s1, s2 = self.init_weight()
        m1_pool_holistic = self.inference_holistic(s1)
        m2_pool_holistic = self.inference_holistic(s2)
        feah_holistic, feab_holistic = self.cal_holistic(m1_pool_holistic, m2_pool_holistic)
        m1_pool_perspective = self.inference_perspective(s1)
        m2_pool_perspective = self.inference_perspective(s2)
        feah_perspective, feab_perspective = self.cal_perspective(m1_pool_perspective, m2_pool_perspective)
        self.temp_concat = tf.concat([feah_holistic,feab_holistic, feah_perspective, feab_perspective], axis=1)
        self.add_dropout()
        self.add_output()
        self.add_loss_acc()

    def init_weight(self):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            _, self.word_embedding = build_glove_dic()
            self.embedding_size = self.word_embedding.shape[1]
            W = tf.get_variable(name='word_embedding', shape=self.word_embedding.shape, dtype=tf.float32,
                                     initializer=tf.constant_initializer(self.word_embedding), trainable=True)
            s1 = tf.nn.embedding_lookup(W, self.input_s1)
            s2 = tf.nn.embedding_lookup(W, self.input_s2)
            s1_expand = tf.expand_dims(s1, -1)
            s2_expand = tf.expand_dims(s2, -1)
            return s1_expand, s2_expand


    def inference_holistic(self, sent):
        m_pool = []
        for i,pool in enumerate(self.h_pool_type):
            ws_pool = []
            for j, ws in enumerate(self.h_ws_sizes):
                with tf.name_scope("holistic-conv-{}pool-{}".format(pool, ws)):
                    # conv layers
                    filter_shape = [ws, self.embedding_size, 1, self.filter_size]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    conv_s1 = tf.nn.conv2d(sent, W, strides=[1, 1, 1, 1],
                                           padding="VALID", name="conv")
                    # pool layers
                    ksize = [1, self.sequence_length - ws + 1, 1, 1]
                    if pool == 'max':
                        pool_out = tf.nn.max_pool(conv_s1, ksize=ksize, strides=[1, 1, 1, 1],
                                                  padding='VALID', name="pool")
                    elif pool == 'mean':
                        pool_out = tf.nn.avg_pool(conv_s1, ksize, strides=[1, 1, 1, 1],
                                                  padding='VALID', name='pool')
                    else:
                        pool_out = -tf.nn.max_pool(-conv_s1, ksize, strides=[1,1,1,1],
                                                  padding='VALID', name="pool")
                    pool_out_expand = tf.expand_dims(tf.squeeze(pool_out, [1,2]), -1)
                    ws_pool.append(pool_out_expand)
            m_pool.append(ws_pool)
        return tf.concat(m_pool, 3)


    def cal_holistic(self, m1_pool, m2_pool):
        with tf.name_scope("cal_holistic"):
            split1_max, split1_mean, split1_min = [tf.squeeze(sp, [0]) for sp in tf.split(m1_pool, 3, 0)]
            split2_max, split2_mean, split2_min = [tf.squeeze(sp, [0]) for sp in tf.split(m2_pool, 3, 0)]
            # calculate heah holistically
            cal_feah_max = tf.reduce_sum(tf.multiply(split1_max, split2_max, name="cal_feah_max"), axis=2)
            cal_feah_mean = tf.reduce_sum(tf.multiply(split1_mean, split2_mean, name="cal_feah_mean"), axis=2)
            cal_feah_min = tf.reduce_sum(tf.multiply(split1_min, split2_min, name="cal_feah_min"), axis=2)
            feah_holistic = tf.concat([cal_feah_max, cal_feah_mean, cal_feah_min], axis=1, name="feah_concat")
            # calculate heab holistically
            cal_feab_max = tf.reduce_sum(tf.multiply(split1_max, split2_max, name="cal_feab_max"), axis=1)
            cal_feab_mean = tf.reduce_sum(tf.multiply(split1_mean, split2_mean, name="cal_feab_mean"), axis=1)
            cal_feab_min = tf.reduce_sum(tf.multiply(split1_min, split2_min, name="cal_feab_min"), axis=1)
            feab_holistic = tf.concat([cal_feab_max, cal_feab_mean, cal_feab_min], axis=1, name="feab_concat")
            return feah_holistic, feab_holistic

    def inference_perspective(self, sent):
        m_pool = []
        for i, pool in enumerate(self.p_pool_type):
            ws_pool = []
            for j, ws in enumerate(self.p_ws_sizes):
                with tf.name_scope("perspective-conv-{}pool-{}".format(pool, ws)):
                    # conv layers
                    filter_shape = [ws, 1, 1, self.filter_size]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    conv_s1 = tf.nn.conv2d(sent, W, strides=[1, 1, 1, 1],
                                           padding="VALID", name="conv")
                    # pool layers
                    ksize = [1, self.sequence_length - ws + 1, 1, 1]
                    if pool == 'max':
                        pool_out = tf.nn.max_pool(conv_s1, ksize=ksize, strides=[1, 1, 1, 1],
                                                  padding='VALID', name="pool")
                    else:
                        pool_out = -tf.nn.max_pool(-conv_s1, ksize, strides=[1, 1, 1, 1],
                                                   padding='VALID', name="pool")
                    pool_out = tf.transpose(tf.squeeze(pool_out, [1]), [0,2,1])
                    ws_pool.append(pool_out)
            m_pool.append(ws_pool)
        return tf.concat(m_pool, 3)

    def cal_perspective(self, m1_pool, m2_pool):
        with tf.name_scope("cal_perspective"):
            split1_max, split1_min = [tf.squeeze(sp, [0]) for sp in tf.split(m1_pool, 2)]
            split2_max, split2_min = [tf.squeeze(sp, [0]) for sp in tf.split(m2_pool, 2)]
            # calculate heah holistically
            cal_feah_max = tf.reduce_sum(tf.multiply(split1_max, split2_max, name="cal_feah_max"), axis=2)
            cal_feah_min = tf.reduce_sum(tf.multiply(split1_min, split2_min, name="cal_feah_min"), axis=2)
            feah_holistic = tf.concat([cal_feah_max, cal_feah_min], axis=1, name="feah_concat")
            # calculate heab holistically
            cal_feab_max = tf.reduce_sum(tf.multiply(split1_max, split2_max, name="cal_feab_max"), axis=1)
            cal_feab_min = tf.reduce_sum(tf.multiply(split1_min, split2_min, name="cal_feab_min"), axis=1)
            feab_holistic = tf.concat([cal_feab_max, cal_feab_min], axis=1, name="feab_concat")
            return feah_holistic, feab_holistic

    def add_dropout(self):
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.temp_concat, self.dropout_keep_prob)

    def add_output(self):
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.temp_concat.shape[1], self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    def add_loss_acc(self):
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.square(self.scores - self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope("pearson"):
            mid1 = tf.reduce_mean(self.scores * self.input_y) - \
                        tf.reduce_mean(self.scores) * tf.reduce_mean(self.input_y)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(self.scores)) - tf.square(tf.reduce_mean(self.scores))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(self.input_y)) - tf.square(tf.reduce_mean(self.input_y)))
            self.pearson = mid1 / mid2

if __name__ == '__main__':
    cnn = TextMPCNN(
        sequence_length=36,
        num_classes=1,
        filter_size=300,
        h_pool_type=['max','mean','min'],
        p_pool_type=['max','min'],
        h_ws_sizes=[1,2,36],
        p_ws_sizes=[1,2],
        l2_reg_lambda=1)