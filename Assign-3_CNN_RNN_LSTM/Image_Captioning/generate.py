import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

import tensorflow.python.platform
from keras.preprocessing import sequence
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
model_path = './models/tensorflow'
feature_path = './feats_train.npy'
annotation_path = './Image_Caption_data/6/small_data.txt'
dim_embed = 256
dim_hidden = 256
dim_in = 4096
batch_size = 1
learning_rate = 0.001
momentum = 0.9
n_epochs = 25


def get_data(annotation_path, feature_path):
     annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
     return np.load(feature_path,'r'), annotations['caption'].values
feats, captions = get_data(annotation_path, feature_path)
print(feats.shape)
print(captions.shape)
print(captions[0])


class Caption_Generator():
    def __init__(self, dim_in, dim_embed, dim_hidden, batch_size, n_lstm_steps, n_words, init_b=None):

        self.dim_in = dim_in
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_words = n_words
        

        with tf.device("/cpu:0"):
            self.word_embedding = tf.Variable(tf.random_uniform([self.n_words, self.dim_embed], -0.1, 0.1), name='word_embedding')

        self.embedding_bias = tf.Variable(tf.zeros([dim_embed]), name='embedding_bias')
        

        self.lstm = tf.contrib.rnn.BasicLSTMCell(dim_hidden)
        

        self.img_embedding = tf.Variable(tf.random_uniform([dim_in, dim_hidden], -0.1, 0.1), name='img_embedding')
        self.img_embedding_bias = tf.Variable(tf.zeros([dim_hidden]), name='img_embedding_bias')

        self.word_encoding = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='word_encoding')
        
        if init_b is not None:
            self.word_encoding_bias = tf.Variable(init_b, name='word_encoding_bias')
        else:
            self.word_encoding_bias = tf.Variable(tf.zeros([n_words]), name='word_encoding_bias')

    def build_model(self): 
        img = tf.placeholder(tf.float32, [self.batch_size, self.dim_in])
        caption_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])
        
        # getting an initial LSTM embedding from our image_imbedding
        image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias
        
        # setting initial state of our LSTM
        state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)

        total_loss = 0.0
        with tf.variable_scope("RNN"):
            for i in range(self.n_lstm_steps): 
                if i > 0:
                    with tf.device("/cpu:0"):
                        current_embedding = tf.nn.embedding_lookup(self.word_embedding, caption_placeholder[:,i-1]) + self.embedding_bias
                else:
                    current_embedding = image_embedding
                if i > 0: 
                    tf.get_variable_scope().reuse_variables()

                out, state = self.lstm(current_embedding, state)

                
                if i > 0:
                    labels = tf.expand_dims(caption_placeholder[:, i], 1)
                    ix_range=tf.range(0, self.batch_size, 1)
                    ixs = tf.expand_dims(ix_range, 1)
                    concat = tf.concat([ixs, labels],1)
                    onehot = tf.sparse_to_dense(
                            concat, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)
                    logit = tf.matmul(out, self.word_encoding) + self.word_encoding_bias
                    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=onehot)
                    xentropy = xentropy * mask[:,i]
                    loss = tf.reduce_sum(xentropy)
                    total_loss += loss

            total_loss = total_loss / tf.reduce_sum(mask[:,1:])
            return total_loss, img,  caption_placeholder, mask


    def build_generator(self, maxlen, batchsize=1):
        img = tf.placeholder(tf.float32, [self.batch_size, self.dim_in])
        image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias
        state = self.lstm.zero_state(batchsize,dtype=tf.float32)

        #declare list to hold the words of our generated captions
        all_words = []
        with tf.variable_scope("RNN"):

            output, state = self.lstm(image_embedding, state)
            previous_word = tf.nn.embedding_lookup(self.word_embedding, [0]) + self.embedding_bias

            for i in range(maxlen):
                tf.get_variable_scope().reuse_variables()

                out, state = self.lstm(previous_word, state)


                # get a get maximum probability word and it's encoding from the output of the LSTM
                logit = tf.matmul(out, self.word_encoding) + self.word_encoding_bias
                best_word = tf.argmax(logit, 1)

                with tf.device("/cpu:0"):
                    # get the embedding of the best_word to use as input to the next iteration of our LSTM 
                    previous_word = tf.nn.embedding_lookup(self.word_embedding, best_word)

                previous_word += self.embedding_bias

                all_words.append(best_word)

        return img, all_words

ixtoword = np.load('data/ixtoword.npy').tolist()
n_words = len(ixtoword)
maxlen=15

tf.reset_default_graph()
sess = tf.InteractiveSession()

caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words)

image, generated_words = caption_generator.build_generator(maxlen=maxlen)


def test(sess, image, generated_words, ixtoword, idx=0): # Naive greedy search

    feats, captions = get_data(annotation_path, feature_path)
    feat = np.array([feats[idx]])
    
    saver = tf.train.Saver()
    sanity_check= False
    saved_path=tf.train.latest_checkpoint(model_path)
    saver.restore(sess, saved_path)
    tf.global_variables_initializer().run()
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0

    n = feats.shape[0]
    for i in range(int(n/5)):
        feat = np.array([feats[i*5]])
        generated_word_index = sess.run(generated_words, feed_dict={image: feat})
        generated_word_index = np.hstack(generated_word_index)

        generated_sentence = [ixtoword[x] for x in generated_word_index]
        print(generated_sentence)

        # computing the BLEU Score
        reference = [captions[i], captions[i+1], captions[i+2], captions[i+3], captions[i+4]]
        b1 += sentence_bleu(reference, generated_sentence, weights=(1, 0, 0, 0))
        b2 += sentence_bleu(reference, generated_sentence, weights=(0.5, 0.5, 0, 0))
        b3 += sentence_bleu(reference, generated_sentence, weights=(0.33, 0.33, 0.33, 0))
        b4 += sentence_bleu(reference, generated_sentence, weights=(0.25, 0.25, 0.25, 0.25))

    b1 = 5*b1/n
    b2 = 5*b2/n
    b3 = 5*b3/n
    b4 = 5*b4/n
    print('b1:', b1, 'b2:', b2, 'b3:', b3, 'b4:', b4)

test(sess, image, generated_words, ixtoword, 1)

