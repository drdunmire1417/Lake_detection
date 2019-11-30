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
#S1_files = sorted(glob("Data/*_S1*7_arctic.tif"))
#S2_files = sorted(glob("Data/*_S2*7_arctic.tif"))
#subsurface_geoms = sorted(glob('Data/shape_files/*4_subsurface.geojson'))
#
#print(S1_files, S2_files, subsurface_geoms)
#
#preprocess = Preprocessor.Preprocessor()
#data, surface_masks, subsurface_masks = preprocess.preprocessData(S1_files, S2_files, None)


'''
LABELLING:
'''

#files = sorted(glob("Data/masks/*.tif"))
#
#label = Labeler.Labeler()
#for i in range(len(surface_masks)):
#    surface, subsurface, ice, subsurface_m, surface_m = label.splitData(data[i], surface_masks[i], subsurface_masks[i])
#    print(i, ": done")
#    
##label.QC(subsurface, subsurface_m, 2)
#label.QC(surface, surface_m, 1)
##label.iceQC(ice)   

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
#train_X, valid_X, test_X, train_label, valid_label, test_label  = train.splitTrainVal(all_data, labels)

'''
MODELING
'''
#model = Model.Model()
#lake_model, lake_model_train = model.train(train_X,train_label, valid_X, valid_label)
#
#model.plotModelStats(lake_model_train)
#model.analyzeAccuracy(lake_model, test_X, test_label)

'''
PREDICTING
'''

#S1_files = sorted(glob("Data/Test/*S1*.tif"))
#S2_files = sorted(glob("Data/Test/*S2*.tif"))
#
#predictor = Predictor.Predictor()
#
#lakes = predictor.predictLabels(S1_files, S2_files)

plt.figure()
plt.imshow(lakes[47])