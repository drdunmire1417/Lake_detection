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
        print("normalized data")
        
        #data[np.isnan(data)] = 0
        #data_norm[np.isnan(data_norm)] = 0
        
        return data, data_norm
    
    def doPCA(self, data):
        
        m = data.shape[0]
        n = data.shape[1]
        d = data.shape[2]
        
        #reshape data
        data2 = data.reshape(m*n,d)
        
        #compute covariance matrix
        C = np.cov(np.transpose(data2))
        
        #eigenvalue decomposition
        evals, evecs = np.linalg.eig(C)
        idx = evals.argsort()[::-1]   
        evals = evals[idx]
        evecs = evecs[:,idx]

        #take top eigenvectors that explain 99% of data
        projected_data = np.dot(data2,evecs[:,0:3]);
        projected_data = projected_data.reshape(m,n,3);
        
        return projected_data
    
    def doClustering(self, data):
        data_flat = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        gmm = GaussianMixture(n_components=3, covariance_type='full')
        
        gmm.fit(data_flat)
        
        labels = gmm.predict(data_flat)
        labels = labels.reshape(data.shape[0],data.shape[1])
        
        return labels
    
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
        if data.shape[0]==4000:
            subsurface[subsurface_out[0,2000:6000,2000:6000] != 0] = 1
        else:
            subsurface[subsurface_out[0,:,:] != 0] = 1


        return subsurface
    
    def makeTif(self, PCAdata, labels, surface, subsurface, fileS1):        
        
        '''
        exports PCA data and masks to tif file so I don't need to do this everytime
        :param PCAdata: 3 bands of PCA data
        :param labels: labels mask
        :param surface: surface water mask
        :param subsurface: subsurface water mask
        :param fileS1: sentinel 1 file to get meta data
        :return:
        '''
        
        outfile = 'Data/masks/mask_' + fileS1.split('_')[-2] + '.tif'
        
        
        with rasterio.open(fileS1) as src:
            meta = src.meta
            
        print(PCAdata.shape)
            
        meta.update(count = 6)
        
        
        if PCAdata.shape[0]==4000:
            meta.update(width = 4000)
            meta.update(height = 4000)
            print(meta)
               
        with rasterio.open(outfile, 'w', **meta) as dst:
             dst.write_band(1, PCAdata[:,:,0])
             dst.write_band(2, PCAdata[:,:,1])
             dst.write_band(3, PCAdata[:,:,2])
             dst.write_band(4, labels.astype(float))
             dst.write_band(5, surface)
             dst.write_band(6, subsurface)
             
        return
        
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
            #sub_geom_file = geoms[i]
            
            data, data_norm = self.loadImages(S1file, S2file)
            
            print(i, ": loaded images")
            #PCAdata = self.doPCA(data_std)
            #print(i, ": PCA done")
            #labels = self.doClustering(PCAdata)
            #print(i, ": clustering done")
            surface_mask = self.surfaceMask(data)
            print(i, ": surface mask done")
            #subsurface_mask = self.subsurfaceMask(data, sub_geom_file, S1file)
            #print(i, ": subsurface mask done")
            
            data3 = np.zeros((data.shape[0], data.shape[1], 3))
            data3[:,:,0] = 0.2989 * data_norm[:,:,2] + 0.5870 * data_norm[:,:,1] + 0.1140 * data_norm[:,:,0]
            data3[:,:,1] = data_norm[:,:,4]
            data3[:,:,2] = data_norm[:,:,5]

            self.data_arrays.append(data3)
            self.surface_masks.append(surface_mask)
            self.subsurface_masks.append(np.zeros((data.shape[0], data.shape[1])))
            #self.makeTif(PCAdata, labels, surface_mask, subsurface_mask, S1file)
            #print(i, ": made geotif")
                

        return self.data_arrays, self.surface_masks, self.subsurface_masks