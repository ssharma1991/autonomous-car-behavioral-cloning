import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import math
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
DATA_DIR='/opt/carnd_p3/data/'
RECOVER_DATA_DIR='/home/workspace/CarND-Behavioral-Cloning-P3/RecoverData/'
TRACK2_DATA_DIR='/home/workspace/CarND-Behavioral-Cloning-P3/track2Data/'
VAL_TRAIN_RATIO=.2
DROPOUT_KEEP_PROB=.5
LEARNING_RATE = 1e-4
n_EPOCHS = 15
BATCH_SIZE = 256
    
# Import Data
def get_data():
    data = pd.read_csv(os.path.join(DATA_DIR, 'driving_log.csv')) # data is pandas.DataFrame type datastructure
    Samples = data[['center', 'left', 'right', 'steering']].values
    
    Recoverdata = pd.read_csv(os.path.join(RECOVER_DATA_DIR, 'driving_log.csv')) # data is pandas.DataFrame type datastructure
    Recoverdata.columns = ('center','left','right','steering','t','b','s')
    recoverSamples=Recoverdata[['center', 'left', 'right', 'steering']].values
    Samples=np.concatenate((Samples, recoverSamples))
    
    track2data = pd.read_csv(os.path.join(TRACK2_DATA_DIR, 'driving_log.csv')) # data is pandas.DataFrame type datastructure
    track2data.columns = ('center','left','right','steering','t','b','s')
    track2Samples=track2data[['center', 'left', 'right', 'steering']].values
    Samples=np.concatenate((Samples, track2Samples))
    
    Samples_train, Samples_valid = train_test_split(Samples, test_size=VAL_TRAIN_RATIO, random_state=0)
    return [Samples_train, Samples_valid]

def augment_data(data):
    # Add left and right camera img as centre with steering bias
    CORRECTION=.5
    Samples_train, Samples_valid= data
    
    aug_samples_train=[]
    for sample in Samples_train:
        center, left, right = sample[0:3]
        center_path=os.path.join(DATA_DIR, center.strip())
        left_path=os.path.join(DATA_DIR, left.strip())
        right_path=os.path.join(DATA_DIR, right.strip())
        center_steer=float(sample[3])
        left_steer=center_steer+CORRECTION
        right_steer=center_steer-CORRECTION
        aug_samples_train.append([center_path,center_steer])
        aug_samples_train.append([left_path,left_steer])
        aug_samples_train.append([right_path,right_steer])
        
    aug_samples_valid=[]
    for sample in Samples_valid:
        center = sample[0]
        center_path=os.path.join(DATA_DIR, center.strip())
        center_steer=float(sample[3])
        aug_samples_valid.append([center_path, center_steer])
    
    return aug_samples_train, aug_samples_valid
    
def get_batch(samples, batch_size, isTrainingSet):
    print(len(samples))
    while True:
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]
            img=[]
            steer=[]
            for batch_sample in batch_samples:
                center_image= mpimg.imread(batch_sample[0])
                center_steer= batch_sample[1]
                if (np.random.random()>.5):
                    img.append(center_image)
                    steer.append(center_steer)
                else:
                    img.append(np.fliplr(center_image))
                    steer.append(-center_steer)
            
            X = np.array(img)
            y = np.array(steer)
            yield X, y

# Build Model
def get_model():
    # Model is inspired from NVIDIA's "End to End Learning for Self-Driving Cars" paper
    model = Sequential()
    
    # Normalization layer
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((65,30), (0,0))))
    
    # 5 Convolution and maxpooling Layer
    model.add(Conv2D(24, (5, 5)))
    model.add(MaxPooling2D((2, 5)))
    model.add(Activation('relu'))
    #30x63x24
    model.add(Conv2D(36, (5, 5)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    #13x29x36
    model.add(Conv2D(48, (3, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    #5x13x48
    model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    #
    model.add(Conv2D(80, (3, 3), activation='relu'))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    
    # 5 Fully connected layers
    model.add(Flatten())
    model.add(Dropout(DROPOUT_KEEP_PROB))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model

# Train Model
def train_model(model, data):
    Samples_train, Samples_valid= data
    Train_data = get_batch(Samples_train, BATCH_SIZE, True)
    Valid_data = get_batch(Samples_valid, BATCH_SIZE, False)
    print(len(Samples_train), len(Samples_valid))
    
    # Compile model with adam optimizer and learning rate of .0001
    adam = Adam(LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    
    # Model will save the weights whenever validation loss improves
    checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')

    # Discontinue training when validation loss fails to decrease
    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

    # Train model for 20 epochs and a batch size of 128
    model.fit_generator(Train_data, steps_per_epoch=math.ceil(len(Samples_train)/BATCH_SIZE), 
                        epochs=n_EPOCHS, validation_data=Valid_data, validation_steps=math.ceil(len(Samples_valid)/BATCH_SIZE),
                        callbacks=[checkpoint, earlystop], verbose=1)

data = get_data()
aug_data= augment_data(data)
print("**************DATA IMPORTED**********************\n")
model = get_model()
print("**************MODEL CREATED**********************\n")
train_model(model, aug_data)
