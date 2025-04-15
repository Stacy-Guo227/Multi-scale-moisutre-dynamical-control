import numpy as np
from ML_model_data import *

import tensorflow as tf
from tensorflow.keras import layers, Input, backend, Model, losses, optimizers, models

# Load vae components
encoder, decoder = Load_ML()
# Load normalized datasets
era5_normIVT_ssn_test = Load_Dataset()['x_test']
era5_normIVT_ssn_val   = Load_Dataset()['x_val']
era5_normIVT_ssn_train = Load_Dataset()['x_train']
temp = np.concatenate((era5_normIVT_ssn_train, era5_normIVT_ssn_val), axis=0)
era5_normIVT_ssn_whole = np.concatenate((temp, era5_normIVT_ssn_test), axis=0)
# Get latent vectors
era5_lvIVT_ssn_test  = encoder.predict(era5_normIVT_ssn_test)[2]
era5_lvIVT_ssn_val   = encoder.predict(era5_normIVT_ssn_val)[2]
era5_lvIVT_ssn_train = encoder.predict(era5_normIVT_ssn_train)[2]
era5_lvIVT_ssn_whole = encoder.predict(era5_normIVT_ssn_whole)[2]
lv_ssn_dict = {'ERA5(test)':era5_lvIVT_ssn_test, 
               'ERA5(val)':era5_lvIVT_ssn_val, 
               'ERA5(train)':era5_lvIVT_ssn_train,  
               'ERA5(whole)':era5_lvIVT_ssn_whole}
"""
Due to the inherent random settings of the model optimizer, the latent vectors will vary very slightly every time the vae-model is called.
The slight difference does not affect the analyses in the study, which are reproducible from simply uncommenting the following line of saving a new set of latent vectors.
For identical latent vectors analyzed in the study, please download from the provided data source.
"""
np.save(f'../../TemporaryData/VAE_model/dataset_file/fix_lv_era5.npy', lv_ssn_dict)