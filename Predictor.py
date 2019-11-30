import gdal
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json

import Preprocessor
import Trainer

class Predictor:
    
    def __init__(self):
        self.preprocess = Preprocessor.Preprocessor()
        self.train = Trainer.Trainer()
        
        json_file = open('lake_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights("lake_model.h5")
        print("Loaded model from disk")
 
        # evaluate loaded model on test data
        self.loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        
        self.lakes = []

        return
        
        
    def splitData(self, data):
        data_arrays = []
        
        dx = 300 #each image will be 300x300 pixels
            
        m = data.shape[0]
        n = data.shape[1]
            
        for i in range(0, m, 100):
            for j in range(0,n, 100):
                
                temp_data = data[i:i+dx, j:j+dx,:]
                
                if temp_data[:,:,0].all()==0 or temp_data.shape[0] != 300 or temp_data.shape[1] != 300:
                    t =0
                else:
                    new_d = np.zeros((self.train.newm,self.train.newn,3))
                    for k in range(3):
                        new_d[:,:,k] = self.train.rebin(temp_data[:,:,k], (self.train.newm, self.train.newn))
                    data_arrays.append(new_d)
         
        data_arrays = np.asarray(data_arrays)
        return data_arrays


    
    
    def predictLabels(self, S1files, S2files):
        
        for i in range(len(S1files)):
            S1file = S1files[i]
            S2file = S2files[i]
                       
            data, data_norm = self.preprocess.loadImages(S1file, S2file)
            
            data3 = np.zeros((data.shape[0], data.shape[1], 3))
            data3[:,:,0] = 0.2989 * data_norm[:,:,2] + 0.5870 * data_norm[:,:,1] + 0.1140 * data_norm[:,:,0]
            data3[:,:,1] = data_norm[:,:,4]
            data3[:,:,2] = data_norm[:,:,5]
                    
            data_arrays = self.splitData(data3)
            
    
            predicted_classes = self.loaded_model.predict(data_arrays)
            predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
            
            for i in range(len(predicted_classes)):
                if predicted_classes[i] == 2:
                    self.lakes.append(data_arrays[i,:,:,:])
            
            
        #score = loaded_model.evaluate(train_X, train_label, verbose=0)

               
        return self.lakes
