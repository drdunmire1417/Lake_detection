import numpy as np
import geopandas
import rasterio
from rasterio.mask import mask
from sklearn.mixture import GaussianMixture
import gdal #
import matplotlib.pyplot as plt


class Preprocessor:
    '''
    This class is intended to take in the paths of the various data and perform preliminary processing and data
    organization for the remainder of the routines in the script.
    -PCA
    -preliminary clustering
    '''
    

    def __init__(self):
        self.data_arrays = []
        self.surface_masks = []
        self.subsurface_masks = []
        return
    
    def standardize(self, data):
        data_std = np.zeros(data.shape)
        for i in range(data.shape[2]):
            data_std[:,:,i] = (data[:,:,i] - np.nanmean(data[:,:,i]))/np.nanstd(data[:,:,i])
        return data_std
    
    def normalize(self, data):
        print("normalizing data")
        data_norm = np.zeros(data.shape)
        for i in range(data.shape[2]):
            data_norm[:,:,i] = (data[:,:,i] - np.nanmin(data[:,:,i]))/(np.nanmax(data[:,:,i] - np.nanmin(data[:,:,i])))
        return data_norm
    
    def loadImages(self, S1file, S2file):
        rasterS1 = gdal.Open(S1file)
        rasterS2 = gdal.Open(S2file)
        
        dataS1 = rasterS1.ReadAsArray()
        dataS2 = rasterS2.ReadAsArray()
        
        data = np.zeros((dataS1.shape[1],dataS1.shape[2],6))
        data[:,:,0] = dataS2[3,:,:] #blue
        data[:,:,1] = dataS2[2,:,:] #green
        data[:,:,2] = dataS2[1,:,:] #red
        data[:,:,3] = dataS2[0,:,:] #nir
        data[:,:,4] = dataS1[0,:,:] #HH
        data[:,:,5] = dataS1[1,:,:] #HV
        
        data_norm = self.normalize(data)

        
        #data[np.isnan(data)] = 0
        #data_norm[np.isnan(data_norm)] = 0
        
        return data, data_norm
    
    
    def surfaceMask(self, data):        
        '''
        creates masks for surface and subsurface water areas
        :param data: all 6 bands of data - to mask surface water
        :return: masks for surface
        '''
        #make surface water mask
        NDWI = (data[:,:,0] - data[:,:,2])/(data[:,:,0] + data[:,:,2])
        surface_water = np.zeros((NDWI.shape))
        surface_water[NDWI > 0.15] = 1
        
        return surface_water

    
    def subsurfaceMask(self, data, geom_file, S1file):        
        '''
        creates masks for surface and subsurface water areas
        :param data: all 6 bands of data - to mask surface water
        :param geom_file: file path of shapes for subsurface lakes
        :param S1 file: file path for Sentinel 1 data
        :return: masks for subsurface water
        '''
        
        #make subsurface mask
        shapefile_sub = geopandas.read_file(geom_file)
        subsurface_geom = shapefile_sub.geometry.values
        with rasterio.open(S1file) as src:
            subsurface_out, out_transform = mask(src, subsurface_geom, crop=False)

        
        subsurface = np.zeros((data.shape[0], data.shape[1]))
        subsurface[subsurface_out[0,:,:] != 0] = 1

        return subsurface
        
    def preprocessData(self, S1Paths, S2Paths, geoms):
        '''
        Extract images from globbed path list
        :param S1paths: Sentinel-1 path list
        :param S2paths: Sentinel-2 path list)
        :return:
        '''
        for i in range(len(S1Paths)):
            S1file = S1Paths[i]
            S2file = S2Paths[i]
            sub_geom_file = geoms[i]
            
            data, data_norm = self.loadImages(S1file, S2file)
            
            print(i, ": loaded images")
            surface_mask = self.surfaceMask(data)
            self.surface_masks.append(surface_mask)
            print(i, ": surface mask done")
            
            if sub_geom_file is not None:
                subsurface_mask = self.subsurfaceMask(data, sub_geom_file, S1file)
                self.subsurface_masks.append(subsurface_mask)
                print(i, ": subsurface mask done")
            else:
                self.subsurface_masks.append(np.zeros((data.shape[0], data.shape[1])))

            data3 = np.zeros((data.shape[0], data.shape[1], 3))
            data3[:,:,0] = 0.2989 * data_norm[:,:,2] + 0.5870 * data_norm[:,:,1] + 0.1140 * data_norm[:,:,0]
            data3[:,:,1] = data_norm[:,:,4]
            data3[:,:,2] = data_norm[:,:,5]

            self.data_arrays.append(data3)
                

        return self.data_arrays, self.surface_masks, self.subsurface_masks