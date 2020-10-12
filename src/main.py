#!/usr/bin/env python
# coding: utf-8

# In[42]:


import os
import pandas as pd, numpy as np
from numpy import asarray, zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Dense
from keras.layers import Concatenate, Flatten, Reshape
from keras.layers import Conv1D,MaxPool1D,Conv2D, MaxPool2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from time import time
import pickle
import preprocessor as p
from tqdm.notebook import tqdm
from tqdm.keras import TqdmCallback
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import tensorflow as tf
import re


# In[2]:


def transpose_label_columns(df):
    print("[INFO] transposing labels")
    return pd.get_dummies(df.target)


# In[3]:



# Read recipe inputs
print("[INFO] Reading dataset...")
data = pd.read_csv("training.csv",header=None,engine='python',encoding='latin1')
data.columns=['Sentiment','Id','Date','Query','User','Text']
data.replace({'Sentiment':{'Negative':0,'Positive':4}},inplace=True)

train_data=data.iloc[:,-1].values
train_label=data.iloc[:,0].values
print(data.columns)


# In[4]:


stemmer=SnowballStemmer('english')


# In[5]:


data['Sentiment'].value_counts()


# In[6]:


print("A few negative comments")
data.head(20)


# In[7]:


print("A few positive comments")
data.tail(20)


# In[8]:


def preprocess(train_data):
    
    def tweet_clean(tweet):
        tweet=re.sub(r'@[A-Za-z0-9]+'," ",tweet) ##Removing the usernames
        tweet=re.sub(r'^[A-Za-z0-9.!?]+'," ",tweet) ##Removing digits and punctuations
        tweet=re.sub(r'https?://[A-Za-z0-9./]+'," ",tweet) ## removing links
        tweet=re.sub(r' +'," ",tweet)
        tweet = tweet.lower()
        tweet = re.sub(r"\'s", " ", tweet)
        tweet = re.sub(r"\'ve", " have ", tweet)
        tweet = re.sub(r"can't", "cannot ", tweet)
        tweet = re.sub(r"n't", " not ", tweet)
        tweet = re.sub(r"\'d", " would ", tweet)
        tweet = re.sub(r"\'ll", " will ", tweet)
        tweet = re.sub(r"\'scuse", " excuse ", tweet)
        tweet = tweet.strip(' ')
        tweet = tweet.strip('. .')
        tweet = tweet.replace('.',' ')
        tweet = tweet.replace('-',' ')
        tweet = tweet.replace("’", "'").replace("′", "'").replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")
        tweet = tweet.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")
        tweet = tweet.replace("€", " euro ").replace("'ll", " will")
        tweet = tweet.replace("don't", "do not").replace("didn't", "did not").replace("im","i am").replace("it's", "it is")
        tweet = tweet.replace(",000,000", "m").replace("n't", " not").replace("what's", "what is")
        tweet = tweet.replace(",000", "k").replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")
        tweet = tweet.replace("he's", "he is").replace("she's", "she is").replace("'s", " own")
        tweet = re.sub('\s+', ' ', tweet)
        tweet=tweet.split()
        tweet=[stemmer.stem(word) for word in tweet if word not in stopwords_man]
        tweet=' '.join(word for word in tweet)

        #all_stopwords = stopwords.words('english')
        return tweet.lower()
    tweets_clean = [p.clean(tweet) for tweet in tqdm(train_data)]
    return tweets_clean


# In[9]:


tweets_clean = preprocess(train_data)


# In[10]:


data['Text_clean']=tweets_clean
data['No_of_Words']=[len(text.split()) for text in tqdm(data['Text_clean'])]


# In[11]:


train_label[train_label==4]=1  ##Resetting the labels for positive tweets to 1


# In[12]:


negatives=data['Sentiment']==0
positives=data['Sentiment']==1
print(train_label)


# In[13]:


fig,ax =plt.subplots(nrows=1,ncols=2,figsize=(15,7.5))

sns.countplot(x=data[positives]['No_of_Words'],label='Positive',ax=ax[0])
sns.countplot(x=data[negatives]['No_of_Words'],label='Negative',ax=ax[1])
ax[0].set_title('Number of words for positive comments')
ax[1].set_title('Number of words for negative comments')
plt.tight_layout()
plt.show()


# In[14]:


data['User'].value_counts().head(20) #Top 20 tweeters


# In[15]:


Max_len=np.max([len(sentence) for sentence in tweets_clean])
Max_len


# In[16]:


def get_tokens(x_train,maxlen):
    tok = None
    tokenizerPath = ""
    try:
        print(" [INFO] :: LOADING TOKENIZER.")
        with open(os.path.join(tokenizerPath, "tokenizer_005.pickle"), 'rb') as f:
            tok = pickle.load(f)
    except:
        print(" [INFO] :: TOKENIZER NOT FOUND...")
        tok = Tokenizer()
        print(" [INFO] :: FITTING NEW TOKENIZER...")
        tok.fit_on_texts(list(x_train))
        print(" [INFO] :: SAVING TOKENIZER...")

        # saving tokenizer for future use
        with open(os.path.join(tokenizerPath, "tokenizer_005.pickle"), 'wb') as f:
            pickle.dump(tok, f, protocol=pickle.HIGHEST_PROTOCOL)

    vocab_size = len(tok.word_index) + 1
    x_train = tok.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
    
    return x_train, vocab_size, tok


# In[17]:


maxlen = 300
data_inputs, vocab_size, tok = get_tokens(tweets_clean,maxlen)


# In[100]:


idx=np.random.randint(0,800000,8000)
test_idx=np.concatenate((idx,idx+800000))

X_test=data_inputs[test_idx]
y_test=train_label[test_idx]

X_test=data_inputs[test_idx]
y_test=train_label[test_idx]
X_train=np.delete(data_inputs,test_idx,axis=0)
y_train=np.delete(train_label,test_idx,axis=0)


# In[21]:


def get_glove(tok, vocab_size, dim_length):
    print(" [INFO] :: Loading GLOVE...")
    embeddingPath = ""
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
    for word, i in tqdm(tok.word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # free memory
    del embeddings_index

    return embedding_matrix


# In[92]:


def get_model(max_sentence_length,dim_length,drop,vocab_size,embedding_matrix):
    # define model
    filter_sizes = [2,3,4]
    inputs = Input(shape=(max_sentence_length,), dtype='int32')
    e = Embedding(vocab_size, dim_length, weights=[embedding_matrix], input_length=max_sentence_length, trainable=False)(inputs)
    #reshape = Reshape((max_sentence_length,dim_length,1))(e)
    conv_0 = Conv1D(filters=int(max_sentence_length/2), kernel_size=filter_sizes[0], padding='valid', activation='relu')(e)
    conv_1 = Conv1D(filters=int(max_sentence_length/2), kernel_size=filter_sizes[1], padding='valid', activation='relu')(e)
    conv_2 = Conv1D(filters=int(max_sentence_length/2), kernel_size=filter_sizes[2], padding='valid', activation='relu')(e)

    maxpool_0 = MaxPool1D(pool_size=max_sentence_length - filter_sizes[0] + 1, strides=1, padding='valid')(conv_0)
    maxpool_1 = MaxPool1D(pool_size=max_sentence_length - filter_sizes[1] + 1, strides=1, padding='valid')(conv_1)
    maxpool_2 = MaxPool1D(pool_size=max_sentence_length - filter_sizes[2] + 1, strides=1, padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dense = Dense(units=256, activation='relu')(flatten)
    dropout = Dropout(drop)(dense)
    output = Dense(units=1, activation='sigmoid')(dropout)

    return Model(inputs=inputs, outputs=output)


# In[93]:


dim_length = 300
batch_size = 16
drop = 0.3
epoches = 50


# In[55]:


embedding_matrix = get_glove(tok, vocab_size, dim_length)


# In[94]:


model = get_model(maxlen,dim_length,drop,vocab_size,embedding_matrix)


# In[95]:


model.summary()


# In[96]:


checkpointPath = "models"
checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])


# In[103]:


print(X_train,X_train.shape,type(X_train))
#X_train = X_train[1:]
y_train = y_train.astype(float)
print(y_train,y_train.shape,type(y_train))


# In[ ]:


X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
print(" [INFO] :: TRAINING MODEL...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epoches, verbose=0, callbacks=[TqdmCallback(verbose=1),checkpoint,tensorboard], validation_split=0.2)


# In[ ]:




