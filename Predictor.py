import gdal
import numpy as np
import matplotlib.pyplot as plt

import Preprocessor
import Trainer

class Predictor:
    
    def __init__(self):
        self.preprocess = Preprocessor.Preprocessor()
        self.train = Trainer.Trainer()
        return
    
    def splitData(self, data):
        data_arrays = []
        
        dx = 300 #each image will be 300x300 pixels
            
        m = data.shape[0]
        n = data.shape[1]
            
        for i in range(0, m, 100):
            for j in range(0,n, 100):
                
                temp_data = data[i:i+dx, j:j+dx,:]
                
                if temp_data[:,:,0].any()==0 or temp_data.shape[0] != 300 or temp_data.shape[1] != 300:
                    t =0
                else:
                    new_d = np.zeros((self.train.newm,self.train.newn,3))
                    for i in range(3):
                        new_d[:,:,i] = self.train.rebin(temp_data[:,:,i], (self.train.newm, self.train.newn))
                    data_arrays.append(new_d)
         
        data_arrays = np.asarray(data_arrays)
        return data_arrays    
    
    
    def predictLabels(self, S1file, S2file):
        data, data_norm = self.preprocess.loadImages(S1file, S2file)
        
        data3 = np.zeros((data.shape[0], data.shape[1], 3))
        data3[:,:,0] = 0.2989 * data_norm[:,:,2] + 0.5870 * data_norm[:,:,1] + 0.1140 * data_norm[:,:,0]
        data3[:,:,1] = data_norm[:,:,4]
        data3[:,:,2] = data_norm[:,:,5]
        
        print(data3.shape)
        
        data_arrays = self.splitData(data3)
               
        return data_arrays
