import pandas  as pd
import numpy as np
import matplotlib.pyplot  as plt
import tensorflow as tf
import warnings
warnings.simplefilter("ignore")

train_data_path ="D://duke//project//sudoku//train.csv" 
test_data_path = "D://duke//project//sudoku//test.csv"

# Reading the csv data and loading it to pandas DataFrame
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
print(np.shape(train_data))
print(np.shape(test_data))

def process_data(dataframe):
    df = dataframe.copy() ## makes the copy of incoming dataframe
    labels = df.pop('label') ## pop function removes the label and assign it to labels variable
    return df, labels

train_images, train_labels = process_data(train_data)
test_images = test_data.copy()

train_images = train_images.values.reshape(-1, 28,28, 1) # As images are gray the no of channel is 1
test_images = test_images.values.reshape(-1, 28,28, 1)  # Here -1 will be replaced by no of samples

# Normalizing the data from [0,255] to [0,1]
train_images = train_images / 255
test_images = test_images / 255

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(2500) # Shuffles the dataset

# Lets split the data to train and validation set
train_dataset = train_ds.skip(8000).batch(32) # skips first 8000 datas and takes the remaining  
vali_dataset = train_ds.take(8000).batch(32)

# importing packages for model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model = tf.keras.Sequential()
model.add(Conv2D(8, (3,3), input_shape=[28,28,1]))
model.add(LeakyReLU())
model.add(Conv2D(16, (3,3)))
model.add(LeakyReLU())
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3)))
model.add(LeakyReLU())
model.add(Conv2D(32, (3,3)))
model.add(LeakyReLU())
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3)))
model.add(LeakyReLU())

model.add(GlobalAveragePooling2D())
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))
# Summary of the model
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
history = model.fit(train_dataset, validation_data=vali_dataset, epochs=15)

# Visualizing the training
stats = pd.DataFrame(history.history)
loss  = stats[['loss', 'val_loss']]
acc = stats[['accuracy', 'val_accuracy']]
loss.plot()
acc.plot()
plt.show()

model.save("model1.h5")

