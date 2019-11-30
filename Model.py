# import the necessary packages
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.regularizers import l1
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np


class Model:
    
    def __init__(self):
        self.epochs = 50
        self.init_lr = 1e-3
        self.bs = 32
        self.num_classes = 3
        return
    
    def build(self, width, height, depth, classes):
        print("[INFO] compiling model...")
        # initialize the model along with the input shape to be
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.2))
        
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
 
		# softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
 
		# return the constructed network architecture
        return model
    
    def plotModelStats(self, model):        
        accuracy = model.history['accuracy']
        val_accuracy = model.history['val_accuracy']
        loss = model.history['loss']
        val_loss = model.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        return
    
    def analyzeAccuracy(self, model, test_X, test_Y_one_hot):
        test_Y = np.argmax(np.round(test_Y_one_hot),axis=1)
        test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=1)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])
        
        
        predicted_classes = model.predict(test_X)
        predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
        target_names = ["Class {}".format(i) for i in range(self.num_classes)]
        print(classification_report(test_Y, predicted_classes, target_names=target_names))
    
    def train(self, train_X, train_labels, valid_X, valid_labels):

        print("[INFO] training network...")
        image_dims = train_X[0].shape
        
        model = self.build(width = image_dims[1], height = image_dims[0], depth = image_dims[2], classes = 3)
        opt = Adam(lr = self.init_lr, decay = self.init_lr/self.epochs)
        model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
        
        model_train = model.fit(train_X, train_labels,batch_size=self.bs,epochs=self.epochs,verbose=1,validation_data=(valid_X, valid_labels))

        # serialize model to JSON
        model_json = model.to_json()
        with open("lake_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("lake_model.h5")
        print("[INFO] Saved model to disk")
        
        return model, model_train
