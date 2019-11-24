import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class Trainer:
    
    def __init__(self):
        
        return
    
    def splitTrainVal(self, train_X, train_Y_one_hot):
        train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
        return train_X, valid_X, train_label, valid_label
    
    def rebin(self, arr, new_shape):
        """Rebin 2D array arr to shape new_shape by averaging."""
        shape = (new_shape[0], arr.shape[0] // new_shape[0], new_shape[1], arr.shape[1] // new_shape[1])
        return arr.reshape(shape).mean(-1).mean(1)

    def jpegToArr(self, image):
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0],3))
        return im_arr
        
    def loadData(self, ice_files, surface_files, sub_files):
        all_data = []
        labels = []
        
        for im in ice_files:
            image = Image.open(im)
            im_arr = self.jpegToArr(image)
            new_im = np.zeros((50,50,3))
            for i in range(3):
                new_im[:,:,i] = self.rebin(im_arr[:,:,i], (50,50))
            all_data.append(new_im)
            labels.append(0)
        
        for im in surface_files:
            image = Image.open(im)
            image_90 = image.transpose(Image.ROTATE_90)
            image_180 = image.transpose(Image.ROTATE_180)
            image_270 = image.transpose(Image.ROTATE_270)
            im_arr = self.jpegToArr(image)
            im_arr_90 = self.jpegToArr(image_90)
            im_arr_180 = self.jpegToArr(image_180)
            im_arr_270 = self.jpegToArr(image_270)
            new_im, new_im90, new_im180, new_im270 = np.zeros((50,50,3)), np.zeros((50,50,3)), np.zeros((50,50,3)), np.zeros((50,50,3))
            for i in range(3):
                new_im[:,:,i] = self.rebin(im_arr[:,:,i], (50,50))
                new_im90[:,:,i] = self.rebin(im_arr_90[:,:,i], (50,50))
                new_im180[:,:,i] = self.rebin(im_arr_180[:,:,i], (50,50))
                new_im270[:,:,i] = self.rebin(im_arr_270[:,:,i], (50,50))
            all_data.append(new_im)
            labels.append(1)
            all_data.append(new_im90)
            labels.append(1)
            all_data.append(new_im180)
            labels.append(1)
            all_data.append(new_im270)
            labels.append(1)

        for im in sub_files:
            image = Image.open(im)
            image_90 = image.transpose(Image.ROTATE_90)
            image_180 = image.transpose(Image.ROTATE_180)
            image_270 = image.transpose(Image.ROTATE_270)
            im_arr = self.jpegToArr(image)
            im_arr_90 = self.jpegToArr(image_90)
            im_arr_180 = self.jpegToArr(image_180)
            im_arr_270 = self.jpegToArr(image_270)
            new_im, new_im90, new_im180, new_im270 = np.zeros((50,50,3)), np.zeros((50,50,3)), np.zeros((50,50,3)), np.zeros((50,50,3))
            for i in range(3):
                new_im[:,:,i] = self.rebin(im_arr[:,:,i], (50,50))
                new_im90[:,:,i] = self.rebin(im_arr_90[:,:,i], (50,50))
                new_im180[:,:,i] = self.rebin(im_arr_180[:,:,i], (50,50))
                new_im270[:,:,i] = self.rebin(im_arr_270[:,:,i], (50,50))
            
            all_data.append(new_im)
            labels.append(2)
            all_data.append(new_im90)
            labels.append(2)
            all_data.append(new_im180)
            labels.append(2)
            all_data.append(new_im270)
            labels.append(2)
    
        all_data = np.asarray(all_data)
        labels = np.asarray(labels)

        return all_data/255, to_categorical(labels)
            
        