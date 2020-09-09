# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from numpy import asarray, zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Dense
from keras.layers import Concatenate, Flatten, Reshape
from keras.layers import Conv2D, MaxPool2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from time import time
import pickle

def drop_null(df): 
    return df.dropna()
    
def drop_neutral(df):
    return df[df["sentiment"]!="Neutral"]


def transpose_label_columns(df):
    
    return pd.get_dummies(df.sentiment)

    
def read_dataset():
    # Read recipe inputs
    master_English_Social_Media_stacked = dataiku.Dataset("Master_English_Social_Media_stacked")
    master_English_Social_Media_stacked_df = master_English_Social_Media_stacked.get_dataframe()
    
    master_English_Social_Media_stacked_df = drop_null(master_English_Social_Media_stacked_df)
    master_English_Social_Media_stacked_df = drop_neutral(master_English_Social_Media_stacked_df)
    
    y = transpose_label_columns(master_English_Social_Media_stacked_df)
    x = master_English_Social_Media_stacked_df["sentence"].values
    
    return x,y

def get_tokens(x_train,maxlen):
    tok = None
    handle = dataiku.Folder("glove")
    tokenizerPath = handle.get_path()

    try:
        print(" [INFO] :: LOADING TOKENIZER.")
        with open(os.path.join(tokenizerPath, "tokenizer_001.pickle"), 'rb') as f:
            tok = pickle.load(f)
    except:
        print(" [INFO] :: TOKENIZER NOT FOUND...")
        tok = Tokenizer()
        print(" [INFO] :: FITTING NEW TOKENIZER...")
        tok.fit_on_texts(list(X_train))
        print(" [INFO] :: SAVING TOKENIZER...")

        # saving tokenizer for future use
        with open(os.path.join(tokenizerPath, "tokenizer_001.pickle"), 'wb') as f:
            pickle.dump(tok, f, protocol=pickle.HIGHEST_PROTOCOL)

    vocab_size = len(tok.word_index) + 1
    x_train = tok.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
    
    return x_train, vocab_size, tok

def get_glove(tok, vocab_size, dim_length):
    print(" [INFO] :: Loading GLOVE...")
    handle = dataiku.Folder("glove")
    embeddingPath = handle.get_path()
    embeddings_index = dict()

    try:
        with open(os.path.join(embeddingPath,"glove.42B.300d.txt"),'r',errors = 'ignore', encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = ''.join(values[:-300])
                coefs = np.asarray(values[-300:], dtype='float32')
                embeddings_index[word] = coefs

    except Exception as e:
        print(' [ERROR] :: GLOVE path not found...',e)
        exit(1)
    
    print(' [INFO] :: Loaded %s word vectors.' % len(embeddings_index))
    
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, dim_length))
    for word, i in tok.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # free memory
    del embeddings_index

    return embedding_matrix

def get_model(max_sentence_length,dim_length,drop,vocab_size,embedding_matrix):
    # define model
    filter_sizes = [3,4,5]
    inputs = Input(shape=(max_sentence_length,), dtype='int32')
    e = Embedding(vocab_size, dim_length, weights=[embedding_matrix], input_length=max_sentence_length, trainable=False)(inputs)
    reshape = Reshape((max_sentence_length,dim_length,1))(e)
    conv_0 = Conv2D(int(max_sentence_length/2), kernel_size=(filter_sizes[0], dim_length), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(int(max_sentence_length/2), kernel_size=(filter_sizes[1], dim_length), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(int(max_sentence_length/2), kernel_size=(filter_sizes[2], dim_length), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(max_sentence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(max_sentence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(max_sentence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=2, activation='sigmoid')(dropout)

    return Model(inputs=inputs, outputs=output)



def train(model, x, y, batch_size, epoches):
    #save trained model checkpoints
    handle = dataiku.Folder("glove")
    checkpointPath = handle.get_path()
    #checkpoint = ModelCheckpoint(os.path.join(checkpointPath,'/weights.{epoch:03d}-{val_acc:.4f}.hdf5'), monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    # Train Model
    print(" [INFO] :: TRAINING MODEL...")
    #model.fit(x_train, y_train, batch_size=batch_size, epochs=epoches, verbose=1, callbacks=[checkpoint,tensorboard], validation_split=0.2)
    model.fit(x, y, batch_size=batch_size, epochs=epoches, verbose=1, validation_split=0.2)
    
    model.save(os.path.join(checkpointPath,'my_model.h5'))

if __name__ == "__main__":
    
    max_sentence_length = 300
    dim_length = 300
    batch_size = 64
    drop = 0.5
    epoches = 50

    x,y = read_dataset()
    x, vocab_size, tok = get_tokens(x,max_sentence_length)
    print(x)
    print("x : ",x.shape)
    print("y : ",y.shape)
    print(type(x),type(x[0]))
       
    #get 42B 300D GLOVE embeddings
    embedding_matrix = get_glove(tok, vocab_size, dim_length)
    
    #get defined model
    model = get_model(max_sentence_length,dim_length,drop,vocab_size,embedding_matrix)
    

    train(model, x, y, batch_size, epoches)
