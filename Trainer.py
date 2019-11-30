import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random

class Trainer:
    
    def __init__(self):
        self.newm = 50
        self.newn = 50
        self.all_data = []
        self.labels = []
        return
    
    def splitTrainVal(self, X, y):
        train_X, test_X, train_labels, test_labels = train_test_split(X, y, test_size=0.15, random_state=1)
        train_X, val_X, train_labels, val_labels = train_test_split(train_X, train_labels, test_size=0.2, random_state=1)        
        
        return train_X, val_X, test_X, train_labels, val_labels, test_labels
    
    def rebin(self, arr, new_shape):
        """Rebin 2D array arr to shape new_shape by averaging."""
        shape = (new_shape[0], arr.shape[0] // new_shape[0], new_shape[1], arr.shape[1] // new_shape[1])
        return arr.reshape(shape).mean(-1).mean(1)

    def jpegToArr(self, image):
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0],3))
        return im_arr
        
    def loadData(self, ice_files, surface_files, sub_files):
        random.shuffle(ice_files)
        random.shuffle(surface_files)
        random.shuffle(sub_files)
        #ice
        for im in ice_files:
            image = Image.open(im)
            im_arr = self.jpegToArr(image)
            new_im = np.zeros((self.newm,self.newn,3))
            for i in range(3):
                new_im[:,:,i] = self.rebin(im_arr[:,:,i], (self.newm,self.newn))
            self.all_data.append(new_im)
            self.labels.append(0)
        #surface lakes
        for im in surface_files:
            image = Image.open(im)
            image_90 = image.transpose(Image.ROTATE_90)
            image_180 = image.transpose(Image.ROTATE_180)
            image_270 = image.transpose(Image.ROTATE_270)
            im_arr = self.jpegToArr(image)
            im_arr_90 = self.jpegToArr(image_90)
            im_arr_180 = self.jpegToArr(image_180)
            im_arr_270 = self.jpegToArr(image_270)
            new_im, new_im90, new_im180, new_im270 = np.zeros((self.newm,self.newn,3)), np.zeros((self.newm,self.newn,3)), np.zeros((self.newm,self.newn,3)), np.zeros((self.newm,self.newn,3))
            for i in range(3):
                new_im[:,:,i] = self.rebin(im_arr[:,:,i], (self.newm,self.newn))
                new_im90[:,:,i] = self.rebin(im_arr_90[:,:,i], (self.newm,self.newn))
                new_im180[:,:,i] = self.rebin(im_arr_180[:,:,i], (self.newm,self.newn))
                new_im270[:,:,i] = self.rebin(im_arr_270[:,:,i], (self.newm,self.newn))
            self.all_data.append(new_im)
            self.labels.append(1)
            self.all_data.append(new_im90)
            self.labels.append(1)
            self.all_data.append(new_im180)
            self.labels.append(1)
            self.all_data.append(new_im270)
            self.labels.append(1)

        #subsurface lakes
        for im in sub_files:
            image = Image.open(im)
            image_90 = image.transpose(Image.ROTATE_90)
            image_180 = image.transpose(Image.ROTATE_180)
            image_270 = image.transpose(Image.ROTATE_270)
            im_arr = self.jpegToArr(image)
            im_arr_90 = self.jpegToArr(image_90)
            im_arr_180 = self.jpegToArr(image_180)
            im_arr_270 = self.jpegToArr(image_270)
            new_im, new_im90, new_im180, new_im270 = np.zeros((self.newm,self.newn,3)), np.zeros((self.newm,self.newn,3)), np.zeros((self.newm,self.newn,3)), np.zeros((self.newm,self.newn,3))
            for i in range(3):
                new_im[:,:,i] = self.rebin(im_arr[:,:,i], (self.newm,self.newn))
                new_im90[:,:,i] = self.rebin(im_arr_90[:,:,i], (self.newm,self.newn))
                new_im180[:,:,i] = self.rebin(im_arr_180[:,:,i], (self.newm,self.newn))
                new_im270[:,:,i] = self.rebin(im_arr_270[:,:,i], (self.newm,self.newn))
            
            self.all_data.append(new_im)
            self.labels.append(2)
            self.all_data.append(new_im90)
            self.labels.append(2)
            self.all_data.append(new_im180)
            self.labels.append(2)
            self.all_data.append(new_im270)
            self.labels.append(2)
    
        self.all_data = np.asarray(self.all_data)
        self.labels = np.asarray(self.labels)

        return self.all_data/255, to_categorical(self.labels)
            
        