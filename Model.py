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
from keras import backend as K


class Model:
    
    def __init__(self):
        self.epochs = 50
        self.init_lr = 1e-3
        self.bs = 32
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
        model.add(Dropout(0.25))
        
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
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
    
    def train(self, train_X, train_labels, valid_X, valid_labels):

        print("[INFO] training network...")
        image_dims = train_X[0].shape
        
        model = self.build(width = image_dims[1], height = image_dims[0], depth = image_dims[2], classes = 3)
        opt = Adam(lr = self.init_lr, decay = self.init_lr/self.epochs)
        model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
        
        model_train = model.fit(train_X, train_labels,batch_size=self.bs,epochs=self.epochs,verbose=1,validation_data=(valid_X, valid_labels))

        return
