from glob import glob
import matplotlib.pyplot as plt
import numpy as np


import Preprocessor
import Labeler 
import Trainer
import Model
import Predictor


'''
PREPROCCESSING: only need to do once
'''
#S1_files = sorted(glob("Data/*_S1*_arctic.tif"))
#S2_files = sorted(glob("Data/*_S2*_arctic.tif"))
#subsurface_geoms = sorted(glob('Data/shape_files/*_subsurface.geojson'))
#
#preprocess = Preprocessor.Preprocessor()
#data, surface_masks, subsurface_masks = preprocess.preprocessData(S1_files, S2_files, subsurface_geoms)

'''
LABELLING:
'''

#files = sorted(glob("Data/masks/*.tif"))
#
#label = Labeler.Labeler()
#for i in range(len(surface_masks)):
#    surface, subsurface, ice, subsurface_m, surface_m = label.splitData(data[i], surface_masks[i], subsurface_masks[i])
#    print(i, ": done")
    
#label.QC(subsurface, subsurface_m, 2)
#label.QC(surface, surface_m, 1)
#label.iceQC(ice)   

'''
TRAINING
'''
#sub_jpgs = glob("Data/jpegs/subsurface/*.jpg")
#surface_jpgs = glob("Data/jpegs/surface/*.jpg")
#ice_jpgs = glob("Data/jpegs/ice/*.jpg")
#
#train = Trainer.Trainer()
#all_data, labels = train.loadData(ice_jpgs, surface_jpgs, sub_jpgs)
#
#train_X, valid_X, train_label, valid_label = train.splitTrainVal(all_data, labels)

'''
MODELING
'''
#model = Model.Model()
#model.train(train_X,train_label, valid_X, valid_label)



'''
PREDICTING
'''

S1_files = sorted(glob("Data/Test/*S1*.tif"))
S2_files = sorted(glob("Data/Test/*S2*.tif"))

predictor = Predictor.Predictor()
lakes = predictor.predictLabels(S1_files, S2_files)

plt.figure()
plt.imshow(lakes[1])