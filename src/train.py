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

import pandas as pd
import pickle

def read(trainPath, testPath):
	print(" [INFO] :: LOADING TRAIN...")
	train_df = pd.read_csv(trainPath, usecols = ["id","comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]).fillna("sterby")
	print(" [INFO] :: LOADING TEST...")
	test_df = pd.read_csv(testPath, usecols = ["id", "comment_text"]).fillna("sterby")
    
	return train_df,test_df

def clean(train_df, test_df):
	print(" [INFO] :: CLEANING TRAIN DATA...")
	X_train = train_df["comment_text"].values
	print(" [INFO] :: CLEANING TRAIN LABELS...")
	y_train = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
	print(" [INFO] :: CLEANING TEST DATA...")
	X_test = test_df["comment_text"].values

	return X_train,X_test, y_train


def get_features(X_train, X_test, maxlen,tokenizerPath):
	tok = None
	
	try:	
		print(" [INFO] :: LOADING TOKENIZER...")
		with open(tokenizerPath, 'rb') as handle:
			tok = pickle.load(handle)
	except:
		print(" [INFO] :: TOKENIZER NOT FOUND...")
		tok = Tokenizer()
		print(" [INFO] :: FITTING NEW TOKENIZER...")
		tok.fit_on_texts(list(X_train) + list(X_test))
		print(" [INFO] :: SAVING TOKENIZER...")
		
		# saving tokenizer for future use
		with open(tokenizerPath, 'wb') as handle:
			pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)

	vocab_size = len(tok.word_index) + 1

	x_train = tok.texts_to_sequences(X_train)
	x_test = tok.texts_to_sequences(X_test)

	x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
	x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')

	return x_train, x_test, vocab_size, tok


def getModel(max_sentence_length,dim_length,drop,vocab_size,embedding_matrix):
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
	output = Dense(units=6, activation='sigmoid')(dropout)
	 
	return Model(inputs=inputs, outputs=output)

def train(isTrain, max_sentence_length, dim_length, epoches, batch_size, drop, modelDirPath, tokenizerPath, trainPath, testPath, embeddingPath, submissionPath, modelPath):
	# load train and test data
	train_df, test_df = read(trainPath,testPath)

	# clean data
	X_train, X_test, y_train = clean(train_df,test_df)

	# get required features and reshape data
	x_train, x_test, vocab_size, tok = get_features(X_train, X_test, max_sentence_length, tokenizerPath)


	#Load embedding vectors
	print(" [INFO] :: Loading GLOVE...")
	embeddings_index = dict()
	
	try:
		f = open(embeddingPath)
		for line in f:
			values = line.split()
			word = values[0]
			coefs = asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
		f.close()
	except:
		print(' [ERROR] :: GLOVE path not found...')
		exit(0)
    
	print(' [INFO] :: Loaded %s word vectors.' % len(embeddings_index))


	# create a weight matrix for words in training docs
	embedding_matrix = zeros((vocab_size, dim_length))
	for word, i in tok.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	# free memory
	del embeddings_index
	del X_train
	del X_test

	#get defined model
	model = getModel(max_sentence_length,dim_length,drop,vocab_size,embedding_matrix)


	if isTrain:	
		#save trained model checkpoints
		checkpoint = ModelCheckpoint(modelDirPath+'/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
		tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

		adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

		# Train Model
		print(" [INFO] :: TRAINING MODEL...")
		model.fit(x_train, y_train, batch_size=batch_size, epochs=epoches, verbose=1, callbacks=[checkpoint,tensorboard], validation_split=0.2) 
	else :
		adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		model.load_weights(modelPath)
		model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
		
		print(" [INFO] :: PREDICTING TESTING DATA...")
		result = model.predict(x_test)

		submission = pd.DataFrame()
		submission["id"] = test_df["id"]
		submission["toxic"] = result[:,0]
		submission["severe_toxic"] = result[:,1]
		submission["obscene"] = result[:,2]
		submission["threat"] = result[:,3]
		submission["insult"] = result[:,4]
		submission["identity_hate"] = result[:,5]

		print(" [INFO] :: SAVING RESULTS FOR TESTING DATA...")
		submission.to_csv(submissionPath,columns = ["id","toxic","severe_toxic","obscene","threat","insult","identity_hate"], index = False)
    
if __name__ == '__main__':
	# parameters

	max_sentence_length = 500
	dim_length = 300
	epoches = 50
	batch_size = 64
	drop = 0.5
	
	modelDirPath = '../trained_models/model_009'
	tokenizerPath = '../tokenizers/tokenizer_009.pickle'
	trainPath = '../training_data/train.csv'
	testPath = '../training_data/test.csv'
	embeddingPath = '../tokenizers/glove.6B.300d.txt'
	submissionPath = '../submission_files/submission_009_test.csv'

	# used only if you want to test your already trained model
	modelPath = '../trained_models/model_009/weights.001-0.9781.hdf5'

	# False if you want to test your model
	isTrain = False

	train(isTrain, max_sentence_length, dim_length, epoches, batch_size, drop, modelDirPath, tokenizerPath, trainPath, testPath, embeddingPath, submissionPath, modelPath)
