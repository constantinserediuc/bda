from __future__ import print_function
from os import environ

environ['PYSPARK_PYTHON'] = '/usr/bin/python3.6'

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
import theano
# theano.config.exception_verbosity='high'

from pyspark.sql import SparkSession

rec = 0
def autoinc(a, b):
    global rec
    rec += 1
    return rec
    # return choices.pop()
    
def add_to_dict(a):
    py_dict = {}
    for line in a:
        x = line.strip("[[\(\)\,\",\'\n").replace("\'", '').replace("\"", '').replace(',', '').split(' ')
        py_dict[x[0]] = int(x[1])
    return py_dict

def tokenize(a, logger):
    a = a.strip("[[\(\)\,\",\'").replace("[", '').replace("]", '').replace("\'", '').replace("\"", '').replace(',', '').split(' ')
    # logger.warn(a)
    tokens = [int(elem) for elem in a]
    return tokens

    
def nn(text, summ, word_dict):
    # Max Length of source sequence
    lenght_list=[]
    for l in text:
        lenght_list.append(len(l))
    max_length_src = np.max(lenght_list)

    # Max Length of target sequence
    lenght_list=[]
    for l in summ:
        lenght_list.append(len(l))
    max_length_tar = np.max(lenght_list)
    # logger.warn(max_length_src)
    # logger.warn(max_length_tar)

    input_words = sorted(list(word_dict))
    target_words = sorted(list(word_dict))
    num_encoder_tokens = int(max(word_dict.values()) + 1)
    num_decoder_tokens = num_encoder_tokens

    # num_decoder_tokens += 1 # For zero padding
    input_token_index = word_dict
    target_token_index = word_dict
    
    reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
    reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())

    # Train - Test Split
    X, y = np.array(text), np.array(summ)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
    # logger.warn(X_train.shape)
    # logger.warn(X_test.shape)

    def generate_batch(X = X_train, y = y_train, batch_size = 128):
        ''' Generate a batch of data '''
        while True:
            for j in range(0, len(X), batch_size):
                encoder_input_data = np.zeros((batch_size, max_length_src),dtype='float32')
                decoder_input_data = np.zeros((batch_size, max_length_tar),dtype='float32')
                decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype='float32')
                for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                    for t, word in enumerate(input_text):
                        encoder_input_data[i, t] = word # encoder input seq
                    for t, word in enumerate(target_text):
                        if t<len(target_text)-1:
                            decoder_input_data[i, t] = word # decoder input seq
                        # if t>0:
                            # decoder target sequence (one hot encoded)
                            # does not include the START_ token
                            # Offset by one timestep
                            # decoder_target_data[i, t - 1, word] = 1.
                yield([encoder_input_data, decoder_input_data], decoder_target_data)
                
    latent_dim = 5


    # Encoder
    encoder_inputs = Input(shape=(None,))
    enc_emb =  Embedding(int(num_encoder_tokens), int(latent_dim), mask_zero = True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = (enc_emb)
    # We discard `encoder_outputs` and only encoder_lstmkeep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
    dec_emb = dec_emb_layer(decoder_inputs)
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='tanh')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    train_samples = len(X_train)
    val_samples = len(X_test)
    batch_size = 1
    epochs = 10
    
    if not os.path.exists("nmt_weights.h5"):
        model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size),
                            steps_per_epoch = train_samples//batch_size,
                            epochs=epochs,
                            validation_data = generate_batch(X_test, y_test, batch_size = batch_size),
                            validation_steps = val_samples//batch_size)
                            
        model.save_weights('nmt_weights.h5')

    # Encode the input sequence to get the "thought vectors"
    else:
        encoder_model = Model(encoder_inputs, encoder_states)

        # Decoder setup
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        dec_emb2= dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

        # Final decoder model
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs2] + decoder_states2)  
        def decode_sequence(input_seq):
            # Encode the input as state vectors.
            states_value = encoder_model.predict(input_seq)
            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1,1))
            # Populate the first character of target sequence with the start character.
            target_seq[0, 0] = 0

            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

                # Sample a token
                # logger.warn(reverse_target_char_index)
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = reverse_target_char_index[sampled_token_index]
                decoded_sentence += ' '+sampled_char

                # Exit condition: either hit max length
                # or find stop character.
                if (sampled_char == '_END' or
                   len(decoded_sentence) > 50):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1,1))
                target_seq[0, 0] = sampled_token_index

                # Update states
                states_value = [h, c]

            return decoded_sentence
            
        train_gen = generate_batch(X_train, y_train, batch_size = 1)
        k=-1
        
        def translate_to_words(idxes):
            sentences = []
            for idx_list in idxes:
                tr = []
                for idx in idx_list:
                    if idx in reverse_input_char_index:
                        tr.append(reverse_input_char_index[idx])
                    sentences.append(tr)
            return sentences
            
        k+=1
        (input_seq, actual_output), _ = next(train_gen)
        decoded_sentence = decode_sequence(input_seq)
        print('\nInput paper words:', translate_to_words(X_train[k:k+1])[0])
        print('\nActual abstract words:', translate_to_words(y_train[k:k+1])[0])
        print('\nPredicted abstract words:', decoded_sentence[:-4])
        
        # k+=1
        # (input_seq, actual_output), _ = next(train_gen)
        # decoded_sentence = decode_sequence(input_seq)
        # print('Input English sentence:', X_train[k:k+1].values[0])
        # print('Actual Marathi Translation:', y_train[k:k+1].values[0][6:-4])
        # print('Predicted Marathi Translation:', decoded_sentence[:-4])
        
        # val_gen = generate_batch(X_test, y_test, batch_size = 1)
        # k=-1
        # k+=1
        # (input_seq, actual_output), _ = next(val_gen)
        # decoded_sentence = decode_sequence(input_seq)
        # print('Input English sentence:', X_test[k:k+1].values[0])
        # print('Actual Marathi Translation:', y_test[k:k+1].values[0][6:-4])
        # print('Predicted Marathi Translation:', decoded_sentence[:-4])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: nn.py <text file> <dict file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .master("local") \
        .appName("WordDict")\
        .getOrCreate()
    logger = logging.getLogger('pyspark')

    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    # counts = lines.map(lambda x: [y.strip(" \)\(\[\]").replace("\"", '').replace("\'", '').replace(',', '') for y in x.split("], [")])
    # counts = counts.map(lambda x: [tokenize(x[0], logger), tokenize(x[1], logger)]).toDF()
    
    counts = lines.map(lambda x: x.split('], ['))
    counts = counts.map(lambda x: [tokenize(x[1], logger), tokenize(x[0], logger)]).toDF()
    # logger.warn((counts.head()))
    # counts.saveAsTextFile("indexes")
    texts = [[int(x) for x in row['_1']] for row in counts.collect()]
    target = [[int(x) for x in row['_2']] for row in counts.collect()]
    # logger.warn(type(counts.head()), type(counts.collect()[0]))
    
    with open('new_dict.json') as data_file:
        dicts = json.load(data_file)
    nn(texts, target, dicts)

    spark.stop()
