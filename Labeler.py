import gdal
import matplotlib.pyplot as plt
import numpy as np
import imageio

class Labeler:
    '''
    This class labels data (subsurface lake, surface lake, ice) for training
    '''
    

    def __init__(self):
        self.ice = []
        self.surface = []
        self.subsurface = []
        
        self.surface_masks = []
        self.subsurface_masks=[]
        
        self.ice_path = 'Data/jpegs/ice/'
        self.surface_path = 'Data/jpegs/surface/'
        self.subsurface_path = 'Data/jpegs/subsurface/'
        
        return
    
    def QC(self, data_list, mask_list, t):
        plt.close('all')
        '''
        Quality check on data: save good ones to jpeg
        param t: type of data, 0 = ice, 1 = surface, 2 = subsurface
        '''
        for i in range(len(data_list)):
            if mask_list is not None:
                fig, axs = plt.subplots(1,2)
                axs[0].imshow(data_list[i])
                axs[1].imshow(mask_list[i])
                plt.show()
                    
            else:
                plt.figure()
                plt.imshow(data_list[i])
                plt.show()
                    
            q = input("Quality: ")
            if q == '1':
                print('Good')
                if t == 0:
                    fid = self.ice_path + 'img' + str(i) + '.jpg'
                    d = data_list[i] * 255
                    imageio.imwrite(fid, d.astype(np.uint8))
                elif t == 1:
                    fid = self.surface_path + 'img' + str(i) + '.jpg'
                    d = data_list[i] * 255
                    imageio.imwrite(fid, d.astype(np.uint8))
                elif t == 2:
                    fid = self.subsurface_path + 'img' + str(i) + '.jpg'
                    d = data_list[i] * 255
                    imageio.imwrite(fid, d.astype(np.uint8))            
        return
       
    def iceQC(self, data):
        count = 0
        for i in range(len(data)):
            if (np.min(data[i]) > 0):
                fid = self.ice_path + 'img' + str(i) + '.jpg'
                d = data[i] * 255
                imageio.imwrite(fid, d.astype(np.uint8))
        return
            
            
    def splitData(self, data, surface_mask, subsurface_mask):
        dx = 300 #each image will be 300x300 pixels
            
        m = data.shape[0]
        n = data.shape[1]
            
        for i in range(0, m, 150):
            for j in range(0,n, 150):
                
                temp_data = data[i:i+dx, j:j+dx,:]
                temp_surface = surface_mask[i:i+dx, j:j+dx]
                temp_subsurface = subsurface_mask[i:i+dx, j:j+dx]
                
                if np.all(temp_data[:,:,0] == 0):
                    t = 1
                else:
                    if np.sum(temp_subsurface) > 2500:
                        self.subsurface.append(temp_data)
                        self.subsurface_masks.append(temp_subsurface)
                    elif np.sum(temp_surface) > 2500:
                        self.surface.append(temp_data)
                        self.surface_masks.append(temp_surface)
                    elif np.sum(temp_subsurface) == 0 and np.sum(temp_surface) == 0:
                        self.ice.append(temp_data)
                
        return self.surface, self.subsurface, self.ice, self.subsurface_masks, self.surface_masks    
            
            
            
            
            