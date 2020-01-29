import tensorflow as tf
import numpy as np
import ee,os
from tensorflow import keras
import datetime
import subprocess


# run on cpu, comment to run on gpu, -1 for cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# enable eager execution
tf.enable_eager_execution()

# set path for calibration and validation data
pathCalibration = r"D:\...\models\urban\data\calibration"
pathValidation = r"D:\...\models\urban\data\validation"
logBands = r"D:\...\models\urban\logs\bands.txt"
logVersion = r"D:\...\models\urban\logs\version.txt"
MODEL_DIR = r"D:\..\models\urban\model"

# List of fixed-length features, all of which are float32.
size = 16

# How many classes there are in the model.
nClasses = 1

# shuffle size and batch size
shuffle_size = 3500
batch_size = 1500
dropout_rate = 0.20

# set class name and labels
label = "class"
bands = ["NBLI","EVI","SAVI","IBI","ND_nir_red",'p20_blue', 'p20_green', 'p20_red', 'p20_nir', 'p20_swir1', 'p20_swir2','blue', 'green', 'red', 'nir', 'swir1', 'swir2','p80_blue', 'p80_green', 'p80_red', 'p80_nir', 'p80_swir1', 'p80_swir2'] #,'jan_blue', 'jan_green', 'jan_red', 'jan_nir', 'jan_swir1', 'jan_swir2', 'apr_blue', 'apr_green', 'apr_red', 'apr_nir', 'apr_swir1', 'apr_swir2', 'jul_blue', 'jul_green', 'jul_red', 'jul_nir', 'jul_swir1', 'jul_swir2', 'oct_blue', 'oct_green', 'oct_red', 'oct_nir', 'oct_swir1', 'oct_swir2']
bands = ["SAVI","p20_SAVI","p80_SAVI","NBLI","EVI","IBI","p20_EVI","p80_EVI","p20_IBI","p80_IBI","p20_NBLI","p80_NBLI", 'p20_blue', 'p20_green', 'p20_red', 'p20_nir', 'p20_swir1', 'p20_swir2','blue', 'green', 'red', 'nir', 'swir1', 'swir2','p80_blue', 'p80_green', 'p80_red', 'p80_nir', 'p80_swir1', 'p80_swir2'] #,'jan_blue', 'jan_green', 'jan_red', 'jan_nir', 'jan_swir1', 'jan_swir2', 'apr_blue', 'apr_green', 'apr_red', 'apr_nir', 'apr_swir1', 'apr_swir2', 'jul_blue', 'jul_green', 'jul_red', 'jul_nir', 'jul_swir1', 'jul_swir2', 'oct_blue', 'oct_green', 'oct_red', 'oct_nir', 'oct_swir1', 'oct_swir2']
bands = sorted(bands)

# get all tensorflow files in directory
def getFiles(path):
	files = []
	# r=root, d=directories, f = files
	for r, d, f in os.walk(path):
		for file in f:
			if '.gz' in file:
				files.append(os.path.join(r, file))
	return files

# get the location of all files
calibrationData = getFiles(pathCalibration)
validationData = getFiles(pathValidation)

## Create a dataset from the TFRecord file in Cloud Storage.
trainDataset = tf.data.TFRecordDataset(calibrationData, compression_type='GZIP')
#testDataset = tf.data.TFRecordDataset(validationData, compression_type='GZIP')


# get length of input array and make list
l = len(bands)
featureNames = list(bands)
featureNames.append(label)

columns = [
  tf.io.FixedLenFeature(shape=[size,size], dtype=tf.float32) for k in featureNames
]

# Dictionary with names as keys, features as values.
featuresDict = dict(zip(featureNames, columns))

# parse data record to record 
def parse_tfrecord(example_proto):
  """The parsing function.

  Read a serialized example into the structure defined by featuresDict.

  Args:
    example_proto: a serialized Example.
  
  Returns: 
    A tuple of the predictors dictionary and the label, cast to an `int32`.
  """
  parsed_features = tf.io.parse_single_example(example_proto, featuresDict)
  labels = parsed_features.pop(label)
  return parsed_features, tf.cast(labels, tf.float32) #return parsed_features, tf.cast(labels, tf.int32)

# Map the function over the dataset.
trainDataset= trainDataset.map(parse_tfrecord)

# get the bands
tup = iter(trainDataset).next()[0]

# store band order to logfile
txt = open(logBands,"w")
for items in tup:
	print(items)
	txt.write(items+"\n")
txt.close()

# Keras requires inputs as a tuple.  Note that the inputs must be in the
# right shape.  Also note that to use the categorical_crossentropy loss,
# the label needs to be turned into a one-hot vector.
def toTuple(dict, label):
  return tf.transpose(list(dict.values())), tf.expand_dims(label,1)

# Repeat the input dataset as many times as necessary in batches.
trainDataset = trainDataset.map(toTuple).shuffle(shuffle_size).batch(batch_size).repeat()

# create the test dataset
testDataset = (
  tf.data.TFRecordDataset(validationData, compression_type='GZIP')
    .map(parse_tfrecord, num_parallel_calls=5)   
    .map(toTuple)  
)


# Define the layers in the model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Input((size, size, l,)),
  tf.keras.layers.Conv2D(32, (1, 1), activation=tf.nn.relu),
  tf.keras.layers.Dropout(dropout_rate),
  tf.keras.layers.Conv2D(16, (1, 1), activation=tf.nn.relu),
  tf.keras.layers.Dropout(dropout_rate),
  tf.keras.layers.Conv2D(8, (1, 1), activation=tf.nn.relu),
  tf.keras.layers.Dropout(dropout_rate),
  tf.keras.layers.Dense(1, activation="linear" )
])

# compile the model
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01),#'adam',
              loss='mse',
              metrics=['mse','mae'])

#set early stop to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30, verbose=0, mode='min')


# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
training = model.fit(x=trainDataset, epochs=500,steps_per_epoch=10,callbacks=[early_stop])

# evaluate the model
evaluate = model.evaluate(testDataset)

print("\n evaluate: \n")
print(evaluate)


# save model
timeStamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
MODEL_DIR = os.path.join(MODEL_DIR, timeStamp)
MODEL_NAME = 'urban_dnn_model_v1'
VERSION_NAME = 'v' + timeStamp
print('Creating version: ' + VERSION_NAME)
PROJECT = "servir-rlcms"
EEIFIED_DIR = MODEL_DIR+ 'eeified'
cloud_DIR = '''gs://servirmekong/model/''' + timeStamp

# store version number in logfile
txt = open(logVersion,"w")
txt.write(VERSION_NAME+"\n")
txt.close()

# create a directory to save the model
os.makedirs(MODEL_DIR)

# store the model
tf.contrib.saved_model.save_keras_model(model, MODEL_DIR) 

# make model ee readable
myString = '''earthengine model prepare --source_dir ''' + MODEL_DIR + ''' --dest_dir  ''' + MODEL_DIR + '''eeified --input "{\\"input_1:0\\":\\"array\\"}" --output "{\\"dense/BiasAdd:0\\":\\"landclass\\"}"'''

# make os call to push model to cloud
subprocess.call(myString,shell=True)
pushData = '''gsutil -m cp -R ''' + EEIFIED_DIR  + " " + cloud_DIR
print(pushData)
subprocess.call(pushData,shell=True)

# push them model from the cloud to AI platform
string = '''gcloud ai-platform versions create ''' + VERSION_NAME+\
  ''' --project ''' + PROJECT+\
  ''' --model ''' + MODEL_NAME+\
  ''' --origin ''' + cloud_DIR+\
  ''' --runtime-version=1.14'''+\
  ''' --framework "TENSORFLOW"'''+\
  ''' --python-version=3.5'''

print(string)
os.system(string)


