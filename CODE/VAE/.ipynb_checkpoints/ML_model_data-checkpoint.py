import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, backend, Model, losses, optimizers, models

latent_dim = 2
model_name = 'vae'

def Load_Dataset():
    """
    Load normalized dataset: {training, validation, testing}
    """
    dataset    = np.load(f'../../TemporaryData/VAE_model/dataset_file/{model_name}.npy', allow_pickle=True).item()
    return dataset

def Load_ML():
    """
    Load saved VAE components: encoder, decoder
    """
    loaded_vae = models.load_model(f'../../TemporaryData/VAE_model/{model_name}.h5')
    # Access the encoder and decoder components
    encoder = loaded_vae.get_layer('encoder')
    decoder = loaded_vae.get_layer('decoder') 
    return encoder, decoder

def Load_latent_vectors(data:str):
    """
    Load saved latent vectors (to avoid the slight random effect from the model optimizer).
    """
    try:
        if data in list(np.load(f'../../TemporaryData/VAE_model/dataset_file/fix_lv_era5.npy', allow_pickle=True).item().keys()):
            return np.load(f'../../TemporaryData/VAE_model/dataset_file/fix_lv_era5.npy', allow_pickle=True).item()[data]
        else:
            print('Wrong data name. Please choose one from below.')
            print(np.load(f'../../TemporaryData/VAE_model/dataset_file/fix_lv_era5.npy', allow_pickle=True).item().keys())
    except:
        print("Please provide the files for saved latent vectors by referring to the code file: fix_latent vectors.py")

def Load_reconstruction(latent_vectors:np.ndarray):
    """
    Decode latent vectors (generate reconstruction)
    """
    encoder, decoder = Load_ML()
    return decoder.predict(latent_vectors)

def Load_datelist():
    import pickle
    with open(f'../../TemporaryData/sum_date_2001_2019.txt', 'rb') as f:
        return pickle.load(f)
    
def Xtest_datelist():
    # Load whole datelist
    date_2001_2019 = Load_datelist()
    # Load dataset
    dataset                = Load_Dataset()
    x_train, x_val, x_test = dataset['x_train'], dataset['x_val'], dataset['x_test']
    train_size, val_size   = x_train.shape[0], x_val.shape[0]
    # Extract x_test datelist
    return date_2001_2019[int(train_size+val_size):]   
