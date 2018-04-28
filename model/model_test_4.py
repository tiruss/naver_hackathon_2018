# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import os

import numpy as np
import tensorflow as tf

import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))

        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):

        left_preprocessed_data, right_preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(output_sigmoid, feed_dict={x_1: left_preprocessed_data, x_2: right_preprocessed_data})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable.left_data[n_idx:min(n_idx + n, length)],\
              iterable.right_data[n_idx:min(n_idx + n , length)], iterable.labels[n_idx:min(n_idx + n, length)]


def Conv_Block(inputs, shortcut, num_filters, name):
    he_normal = tf.keras.initializers.he_normal()
    regularizer = tf.contrib.layers.l2_regularizer(1e-4)

    filter_shape = [5, 5, inputs.get_shape()[3], num_filters]
    weight = tf.get_variable(name=name, shape=filter_shape, initializer=he_normal, regularizer=regularizer)
    inputs = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding="SAME")
    inputs = tf.layers.batch_normalization(inputs=inputs, momentum=0.997, epsilon=1e-5,
                                               center=True, scale=True)
    inputs = tf.nn.relu(inputs)

    if shortcut is not None:
        inputs += shortcut
        return inputs

    return inputs


def downsampling(inputs, shortcut=None):
    print("Shortcut Shape : ", shortcut.get_shape())
    print("Input Shape : ", inputs.get_shape())
    pool = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2,2], strides=2, padding="same")
    print("Pool Shape : ", pool.get_shape())
    shortcut = tf.layers.conv2d(inputs=shortcut, filters=shortcut.get_shape()[3], kernel_size=1,
                                strides=2, padding="same", use_bias=False)
    print("Shortcut Shape : ", shortcut.get_shape())

    pool = pool + shortcut
    print("=============", pool.get_shape())

    return tf.layers.conv2d(inputs=pool, filters=pool.get_shape()[3]*2, kernel_size=1, strides=1,
                            padding="valid", use_bias=False)


def model(inputs, name, embedding_size=16, optional_shortcut=False):
    he_normal = tf.keras.initializers.he_normal()
    regularizer = tf.contrib.layers.l2_regularizer(1e-4)

    layers = []
    num_layers = [10, 10, 4, 4]

    with tf.variable_scope("first_conv") as scope:
        filter_shape = [5, 5, 1, 64]
        weight = tf.get_variable(name=name+"_1", shape=filter_shape, initializer=he_normal,
                                 regularizer=regularizer)
        inputs = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding="SAME")
        print("First Conv : ", inputs.get_shape())
        layers.append(inputs)

    # Conv Block 64
    for a in range(num_layers[0]):
        if a < (num_layers[0] - 1) and optional_shortcut:
            shortcut = layers[-1]
        else:
            shortcut = None
        conv_block = Conv_Block(inputs=layers[-1], shortcut=shortcut, num_filters=64, name=name+"1"+str(1+a))
        layers.append(conv_block)

    pool1 = downsampling(inputs=layers[-1], shortcut=layers[-2])
    layers.append(pool1)

    # Conv Block 128
    for a in range(num_layers[1]):
        if a < num_layers[1] - 1 and optional_shortcut:
            shortcut = layers[-1]
        else:
            shortcut = None
        conv_block = Conv_Block(inputs=layers[-1], shortcut=shortcut, num_filters=128, name=name+"2"+str(a+1))
        layers.append(conv_block)
    pool2 = downsampling(inputs=layers[-1], shortcut=layers[-2])
    layers.append(pool2)
    print("Pooling : ", pool2.get_shape())

    # Conv Block 256
    for a in range(num_layers[2]):
        if a < num_layers[2] - 1 and optional_shortcut:
            shortcut = layers[-1]
        else:
            shortcut = None
        conv_block = Conv_Block(inputs=layers[-1], shortcut=shortcut, num_filters=256, name=name+"3"+str(a+1))
        layers.append(conv_block)
    pool3 = downsampling(inputs=layers[-1], shortcut=layers[-2])
    layers.append(pool3)
    print("Pooling : ", pool3.get_shape())

    # Conv Block 512
    for a in range(num_layers[3]):
        if a < num_layers[3] - 1 and optional_shortcut:
            shortcut = layers[-1]
        else:
            shortcut = None
        conv_block = Conv_Block(inputs=layers[-1], shortcut=shortcut, num_filters=512, name=name+"4"+str(a+1))
        layers.append(conv_block)

    # Extract 8 most features
    # k_pooled = tf.nn.top_k(tf.transpose(layers[-1], [0, 1, 1, 1]), k=8, name="k_pool"+name, sorted=False)[0]
    # print("8_maxpooling: ", k_pooled.get_shape())
    print("---------------- ", layers[-1])
    flatten = tf.reshape(layers[-1], (-1, 2 * 8 * 512))

    return flatten


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch', type=int, default=512)
    args.add_argument('--strmaxlen', type=int, default=64)
    args.add_argument('--embedding', type=int, default=16)
    args.add_argument('--threshold', type=float, default=0.5)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    # 모델의 specification
    input_size = config.embedding*config.strmaxlen # 32 * 256
    learning_rate = 0.000001
    character_size = 256
    strmaxlen = config.strmaxlen
    embedding = config.embedding

    x_1 = tf.placeholder(tf.int32, [None, strmaxlen], name="input_x_1")
    x_2 = tf.placeholder(tf.int32, [None, strmaxlen], name="input_x_2")
    y_ = tf.placeholder(tf.float32, [None, 1], name="input_y")

    # embedding..
    char_embedding = tf.get_variable('char_embedding', [character_size, embedding, 1])
    embedded_1 = tf.nn.embedding_lookup(char_embedding, x_1)
    embedded_2 = tf.nn.embedding_lookup(char_embedding, x_2)

    # create different models..
    model_1 = model(inputs=embedded_1, name="first_W", embedding_size=embedding, optional_shortcut=True)
    model_2 = model(inputs=embedded_2, name="second_W", embedding_size=embedding, optional_shortcut=True)

    # Concatenate 2 model
    Weight7 = tf.Variable(tf.random_normal([8 * 2 * 512 * 2, 2048], stddev=0.01))
    bias_1 = tf.get_variable('bias_1', [2048], initializer=tf.constant_initializer(1.0))
    FC1 = tf.matmul(tf.concat([model_1, model_2], 1), Weight7) + bias_1
    FC1 = tf.nn.relu(FC1)

    Weight8 = tf.Variable(tf.random_normal([2048, 1024], stddev=0.01))
    bias_2 = tf.get_variable('bias_2', [1024], initializer=tf.constant_initializer(1.0))
    FC2 = tf.matmul(FC1, Weight8) + bias_2
    FC2 = tf.nn.relu(FC2)

    Weight9 = tf.Variable(tf.random_normal([1024, 1], stddev=0.01))
    bias_3 = tf.get_variable('bias_3', [1], initializer=tf.constant_initializer(1.0))
    output = tf.matmul(FC2, Weight9) + bias_3
    print("===============", output.get_shape())

    output_sigmoid = tf.nn.sigmoid(output)

    # loss와 optimizer
    # binary_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=output))

    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, output_sigmoid))))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(rmse)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (left_data, right_data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                _, loss = sess.run([train_step, rmse], feed_dict={x_1: left_data, x_2: right_data, y_: labels})
                # _, right_loss = sess.run([train_step, rmse], feed_dict={x: left_data, y_: labels})
                loss = float(loss)

                print('Batch : ', i + 1, '/', one_batch_size, ', RMSE in this minibatch: ', loss)
                avg_loss += loss
            print('epoch:', epoch, ' train_loss:', avg_loss/one_batch_size)
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=avg_loss/one_batch_size, step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)