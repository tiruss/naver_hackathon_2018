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
        pred = sess.run(output, feed_dict={x_1: left_preprocessed_data, x_2: right_preprocessed_data})
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


# Don't use
def conv_block(k_size=5, h_size=64, input=None):
    # Layer1 64 * 32 * 64 with 2 layer
    Weight = tf.Variable(tf.random_normal([k_size, k_size, 1, h_size], stddev=0.01))
    Layer = tf.nn.conv2d(input, Weight, strides=[1, 1, 1, 1], padding="SAME")
    Layer = tf.nn.relu(Layer)
    Layer = tf.layers.batch_normalization(Layer)
    Layer = tf.nn.max_pool(Layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    return Layer


def inception2d(inputs, input_channel, channel_size):

    # bias = tf.Variable(tf.constant(0.1, shape=[channel_size]))

    first_weight = tf.Variable(tf.truncated_normal([1, 1, input_channel, channel_size]))
    first_layer = tf.nn.conv2d(inputs, first_weight, strides=[1, 1, 1, 1], padding="SAME")

    second_weight = tf.Variable(tf.truncated_normal([3, 3, input_channel, channel_size]))
    second_layer = tf.nn.conv2d(inputs, second_weight, strides=[1, 1, 1, 1], padding="SAME")

    third_weight = tf.Variable(tf.truncated_normal([5, 5, input_channel, channel_size]))
    third_layer = tf.nn.conv2d(inputs, third_weight, strides=[1, 1, 1, 1], padding="SAME")

    # pooling = tf.nn.avg_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

    outputs = tf.concat([first_layer, second_layer, third_layer], axis=3)
    # outputs = tf.nn.bias_add(outputs, bias)
    outputs = tf.nn.relu(outputs)

    return outputs


def model(embedded):

    # ====================== Conv Block 64, 128, 256, 512 =======================
    layer1 = inception2d(embedded, 1, 8)
    layer1 = tf.layers.batch_normalization(layer1)
    layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    layer2 = inception2d(layer1, 24, 32)
    layer2 = tf.layers.batch_normalization(layer2)
    layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    layer3 = inception2d(layer2, 96, 128)
    layer3 = tf.layers.batch_normalization(layer3)
    layer3 = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    layer4 = inception2d(layer3, 384, 512)
    layer4 = tf.layers.batch_normalization(layer4)
    layer4 = tf.nn.max_pool(layer4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    layer5 = inception2d(layer4, 1536, 2048)
    layer5 = tf.layers.batch_normalization(layer5)

    # =============== FC Layer 1 ===================
    weight6 = tf.Variable(tf.random_normal([4 * 1 * 512 * 12, 1024], stddev=0.01))
    fc_layer6 = tf.reshape(layer5, [-1, 4 * 1 * 512 * 12])
    fc_layer6 = tf.matmul(fc_layer6, weight6)
    fc_layer6 = tf.nn.relu(fc_layer6)

    return fc_layer6


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=2048)
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

    x_1 = tf.placeholder(tf.int32, [None, strmaxlen])
    x_2 = tf.placeholder(tf.int32, [None, strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, 1])

    # embedding..
    char_embedding = tf.get_variable('char_embedding', [character_size, embedding, 1])
    embedded_1 = tf.nn.embedding_lookup(char_embedding, x_1)
    embedded_2 = tf.nn.embedding_lookup(char_embedding, x_2)

    # create different models..
    model_1 = model(embedded=embedded_1) # 1024 * 1
    model_2 = model(embedded=embedded_2) # 1024 * 1

    # Concatenate 2 model
    Weight7 = tf.Variable(tf.random_normal([1024 * 2, 2048], stddev=0.01))
    FC1 = tf.matmul(tf.concat([model_1, model_2], 1), Weight7)
    FC1 = tf.nn.relu(FC1)

    Weight8 = tf.Variable(tf.random_normal([2048, 1024], stddev=0.01))
    FC2 = tf.matmul(FC1, Weight8)
    FC2 = tf.nn.relu(FC2)

    Weight9 = tf.Variable(tf.random_normal([1024, 1], stddev=0.01))
    output = tf.matmul(FC2, Weight9)

    # output_sigmoid = tf.nn.sigmoid(output)

    # loss와 optimizer
    binary_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=output))
    # rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, output_sigmoid))))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(binary_cross_entropy)

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
                _, loss = sess.run([train_step, binary_cross_entropy], feed_dict={x_1: left_data, x_2: right_data, y_: labels})
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