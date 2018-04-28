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
              iterable.right_data[n_idx:min(n_idx + n , length)], iterable.labels[n_idx:min(n_idx + n , length)]


def model(embedded, name):
    # input_size = embedding * strmaxlen  # 32 * 256
    kernel_size = 5

    # Layer1 64 * 32 * 64 with 2 layer
    weight1 = tf.get_variable(name=name, shape=[kernel_size, kernel_size, 1, 64],
                              initializer=tf.contrib.layers.xavier_initializer())
    # Weight1 = tf.Variable(tf.random_normal([kernel_size, kernel_size, 1, 64], stddev=0.01))
    Layer1 = tf.nn.conv2d(embedded, weight1, strides=[1, 1, 1, 1], padding="SAME")
    Layer1 = tf.nn.relu(Layer1)
    Layer1 = tf.nn.dropout(Layer1, 0.5)
    Layer1 = tf.layers.batch_normalization(Layer1)
    Layer1 = tf.nn.max_pool(Layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Layer2 32 * 16 * 128 with 2 layer
    # Weight2 = tf.Variable(tf.random_normal([kernel_size, kernel_size, 64, 128], stddev=0.01))
    Weight2 = tf.get_variable(name=name+"1", shape=[kernel_size, kernel_size, 64, 128],
                              initializer=tf.contrib.layers.xavier_initializer())
    Layer2 = tf.nn.conv2d(Layer1, Weight2, strides=[1, 1, 1, 1], padding="SAME")
    Layer2 = tf.nn.relu(Layer2)
    Layer2 = tf.nn.dropout(Layer2, 0.5)
    Layer2 = tf.layers.batch_normalization(Layer2)
    Layer2 = tf.nn.max_pool(Layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Layer3 16 * 8 * 256 with 2 layer
    # Weight3 = tf.Variable(tf.random_normal([kernel_size, kernel_size, 128, 256], stddev=0.01))
    Weight3 = tf.get_variable(name=name + "2", shape=[kernel_size, kernel_size, 128, 256],
                              initializer=tf.contrib.layers.xavier_initializer())
    Layer3 = tf.nn.conv2d(Layer2, Weight3, strides=[1, 1, 1, 1], padding="SAME")
    Layer3 = tf.nn.relu(Layer3)
    Layer3 = tf.nn.dropout(Layer3, 0.5)
    Layer3 = tf.layers.batch_normalization(Layer3)
    Layer3 = tf.nn.max_pool(Layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Layer4 8 * 4 * 512 with 2 layer
    # Weight4 = tf.Variable(tf.random_normal([kernel_size, kernel_size, 256, 512], stddev=0.01))
    Weight4 = tf.get_variable(name=name + "3", shape=[kernel_size, kernel_size, 256, 512],
                              initializer=tf.contrib.layers.xavier_initializer())
    Layer4 = tf.nn.conv2d(Layer3, Weight4, strides=[1, 1, 1, 1], padding="SAME")
    Layer4 = tf.nn.relu(Layer4)
    Layer4 = tf.nn.dropout(Layer4, 0.5)
    Layer4 = tf.layers.batch_normalization(Layer4)
    Layer4 = tf.nn.max_pool(Layer4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Weight5 = tf.Variable(tf.random_normal([kernel_size, kernel_size, 512, 512], stddev=0.01))
    Weight5 = tf.get_variable(name=name + "4", shape=[kernel_size, kernel_size, 512, 512],
                              initializer=tf.contrib.layers.xavier_initializer())
    Layer5 = tf.nn.conv2d(Layer4, Weight5, strides=[1, 1, 1, 1], padding="SAME")
    Layer5 = tf.nn.relu(Layer5)
    Layer5 = tf.nn.dropout(Layer5, 0.5)
    Layer5 = tf.layers.batch_normalization(Layer5)

    Weight6 = tf.Variable(tf.random_normal([4 * 1 * 512, 1024], stddev=0.01))
    Layer6 = tf.reshape(Layer5, [-1, 4 * 1 * 512])
    Layer6 = tf.nn.dropout(Layer6, 0.5)
    Layer6 = tf.matmul(Layer6, Weight6)
    Layer6 = tf.nn.relu(Layer6)

    return Layer6


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--batch', type=int, default=512)
    args.add_argument('--strmaxlen', type=int, default=64)
    args.add_argument('--embedding', type=int, default=16)
    args.add_argument('--threshold', type=float, default=0.5)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    # 모델의 specification
    input_size = config.embedding*config.strmaxlen # 32 * 256
    learning_rate = 0.0002
    character_size = 256
    strmaxlen = config.strmaxlen
    embedding = config.embedding

    x_1 = tf.placeholder(tf.int32, [None, strmaxlen])
    x_2 = tf.placeholder(tf.int32, [None, strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, 1])

    # embedding..
    char_embedding = tf.get_variable('char_embedding', [character_size, embedding, 1])
    embedded_1 = tf.nn.embedding_lookup(char_embedding, x_1)
    print(embedded_1)
    embedded_2 = tf.nn.embedding_lookup(char_embedding, x_2)

    # create different models..
    model_1 = model(embedded=embedded_1, name="first_W") # 1024 * 1
    model_2 = model(embedded=embedded_2, name="second_W") # 1024 * 1

    # Concatenate 2 model
    Weight7 = tf.Variable(tf.random_normal([2048, 1024], stddev=0.01))
    FC1 = tf.matmul(tf.concat([model_1, model_2], 1), Weight7)
    FC1 = tf.nn.relu(FC1)

    Weight9 = tf.Variable(tf.random_normal([1024, 1], stddev=0.01))
    output = tf.matmul(FC1, Weight9)

    output_sigmoid = tf.nn.sigmoid(output)

    # loss와 optimizer
    # binary_cross_entropy = tf.reduce_mean(tf.nn.cro(labels=y_, logits=output))
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
                _, loss = sess.run([train_step, rmse],
                                   feed_dict={x_1: left_data, x_2: right_data, y_: labels})
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