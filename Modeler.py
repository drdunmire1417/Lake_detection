import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


class Model:
    
    def __init__(self):
        self.batch_size = 64
        self.epochs = 20
        self.num_classes = 3
        return
    
    def buildModel(self, train_X, train_label, valid_X, valid_label):
        cnn = Sequential()
        cnn.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(50,50,3),padding='same'))
        cnn.add(LeakyReLU(alpha=0.1))
        cnn.add(MaxPooling2D((2, 2),padding='same'))
        cnn.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
        cnn.add(LeakyReLU(alpha=0.1))
        cnn.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        cnn.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
        cnn.add(LeakyReLU(alpha=0.1))                  
        cnn.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        cnn.add(Flatten())
        cnn.add(Dense(128, activation='linear'))
        cnn.add(LeakyReLU(alpha=0.1))                  
        cnn.add(Dense(self.num_classes, activation='softmax'))
        
        cnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        
        lake_train = cnn.fit(train_X, train_label, batch_size=self.batch_size,epochs=self.epochs,verbose=1,validation_data=(valid_X, valid_label))

        
        return


