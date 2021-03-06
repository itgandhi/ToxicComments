# ToxicComments
Kaggle : Toxic Comment Classification Challenge Identify and classify toxic online comments

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them
1. [Python](http://docs.python-guide.org/en/latest/starting/install3/linux/)
2. [NumPy](http://www.numpy.org/)
3. [SciPy](https://www.scipy.org/)
4. [TensorFlow](https://www.tensorflow.org/)
5. [Keras](https://keras.io/)
6. [HDF5](https://support.hdfgroup.org/HDF5/)



## Deployment

1. This project contains 1 src directory which contains 1 file named train.py

2. Important variables need to know.

	#### max_sentence_length = [integer value]
	###### Defines maximum length of the comment you want to keep. eg., 500 then comments greater than 500 characters will be trimmed off and less than 500 will be padded by blank space.
  
	#### dim_length = [integer value]
	###### Defines dimensions of each word. loosely we can say that one word connected to it's nearest words by meaning.
 
	#### batch_size = [integer value]
	###### Training batch size.
  
	#### drop = [real value]
	###### Dropout layers drop value.
	
	#### epoches = [integer value]
	###### For how much iteration or epochs, one want to train the model.
	
	#### modelDirPath = [string value]
	###### A path to a directory where you want to store trained models and checkpoints.
   
	#### tokenizerPath = [string value]
	###### A path to previously tokenized pickle file. if it does not exists it will create a new one.
  
	#### trainPath = [string value]
	###### A path to training data.
	
	#### testPath = [string value]
	###### A path to testing data.
  
	#### embeddingPath = [string value]
	###### A path to GLOVE embedding vector file. select file accordingly to dim_length otherwise it will give you an error.
  
	#### submissionPath = [string value]
	###### A path to store submission file.
	
	#### modelPath = [string value]
	###### Used only if you want to test your already trained model. 

	#### isTrain = [True or False]
	###### True if you want to train your model.
	###### False if you want to test your model.

3. Run `python3 train.py` 


## Authors

* **Ishit Gandhi** - *Initial work* - [IshitGandhi](https://github.com/itgandhi)


## License

This project is Not licensed.
