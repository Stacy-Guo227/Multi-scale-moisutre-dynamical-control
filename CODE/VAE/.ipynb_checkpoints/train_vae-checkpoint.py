# --- Import --- #
import numpy as np
import netCDF4 as nc

import tensorflow as tf
from tensorflow.keras import layers, Input, backend, Model, losses, optimizers, models, callbacks

# Use only one GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Assign season
season     = 'sum'


# --- Function --- #
def read_IVT_1000_700(season:str):
    nctemp = nc.Dataset('../../TemporaryData/IVT_1000_700_2001_2019.nc')
    lat    = nctemp.variables['latitude'][:].data
    lon    = nctemp.variables['longitude'][:].data
    IVT    = nctemp.variables[season+'_IVT'][:].data
    nctemp.close()
    return lat, lon, IVT

## Normalize dataset: each 2D map has a scaler consisting of min and max
def MinMaxNorm_map(arr:np.ndarray):
    ## min/max arrays for each 2D map
    min_arr, max_arr = np.min(arr, axis=(1, 2)), np.max(arr, axis=(1, 2))
    ## expand the arrays into dim:(sample, lat, lon)
    min_arr          = np.repeat(min_arr[:, np.newaxis], arr.shape[1], 1)
    max_arr          = np.repeat(max_arr[:, np.newaxis], arr.shape[1], 1)
    min_arr          = np.repeat(min_arr[..., np.newaxis], arr.shape[2], 2)
    max_arr          = np.repeat(max_arr[..., np.newaxis], arr.shape[2], 2)
    ## calculate normalization
    norm_arr         = (arr-min_arr)/(max_arr-min_arr)
    return norm_arr
    
## Create train, val, test datasets
def create_dataset(season:str, train_size:int, val_size:int):
    lat, lon, x_org = read_IVT_1000_700(season)  # load org. dataset
    x_org   = MinMaxNorm_map(x_org.copy())       # normalize dataset
    x_org   = x_org[:, ::5, ::5, np.newaxis]     # add 'channel' dimension at the last axis
    x_train = x_org[:train_size, ...]
    x_val   = x_org[train_size:train_size+val_size, ...]
    x_test  = x_org[train_size+val_size:, ...]
    return lat, lon, x_train, x_val, x_test
    

# --- Operation --- #
# dimension of the latent space
latent_dim = 2
# Set random seed for reproducibility
seed = 30
tf_seed = tf.random.set_seed(seed)
np.random.seed(seed)
# Set early stopping and save the best model
callback = callbacks.EarlyStopping(monitor='val_loss', patience=400, restore_best_weights=True)

# Input data
season            = 'sum'                      # designate season
input_var         = 'IVT'                      # input variable
train_size        = 2000                       # sample size (how many days)
val_size          = 500
lat, lon, x_train, x_val, x_test = create_dataset(season, train_size, val_size)
channels          = 1                          # (how many variables)
input_dim         = (lat[::5].shape[0], lon[::5].shape[0], channels)    # (lat, lon, channels)



# Encoder
encoder_inputs = Input(shape=input_dim)
x = layers.Conv2D(16, kernel_size=3, activation='relu', padding='same')(encoder_inputs)
x = layers.Conv2D(16, kernel_size=3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
x = layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
x = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
x = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')(x)
x = layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')(x)

x = layers.Flatten()(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

class SamplingLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = backend.random_normal(shape=(backend.shape(z_mean)[0], latent_dim), 
                                        mean=0., stddev=1., seed=tf_seed)
        return z_mean + backend.exp(z_log_var * 0.5) * epsilon

# Replace Lambda with the new SamplingLayer
z = SamplingLayer()([z_mean, z_log_var])

# Decoder
decoder_inputs = Input(shape=(latent_dim,))
x = layers.Dense(4*8*256, activation='relu')(decoder_inputs)

x = layers.Reshape((4,8,256))(x)

x = layers.Conv2DTranspose(256, kernel_size=3, activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(256, kernel_size=3, activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(128, kernel_size=3, activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(128, kernel_size=3, activation='relu', padding='same')(x)
x = layers.UpSampling2D(size=(2, 2))(x)

x = layers.Conv2DTranspose(64, kernel_size=3, activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(64, kernel_size=3, activation='relu', padding='same')(x)
x = layers.UpSampling2D(size=(2, 2))(x)

x = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(32, kernel_size=3, activation='relu')(x)
x = layers.UpSampling2D(size=(2, 2))(x)

x = layers.Conv2DTranspose(16, kernel_size=3, activation='relu')(x)
x = layers.Conv2DTranspose(16, kernel_size=3, activation='relu')(x)
x = layers.Cropping2D(cropping=((2, 1), (1, 0)))(x)
x = layers.Conv2DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')(x)

decoder_outputs = x
# Define the VAE as a Keras Model
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = Model(decoder_inputs, decoder_outputs, name='decoder')

vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = Model(encoder_inputs, vae_outputs, name='vae')
print(encoder.summary())
print(decoder.summary())
print(vae.summary())

# Define the loss function
reconstruction_loss = losses.mean_squared_error(backend.flatten(encoder_inputs), backend.flatten(vae_outputs))
reconstruction_loss*= 1E8    # balancing the recon_loss and KLD_loss
kl_loss             = -0.5 * backend.mean(1 + z_log_var - backend.square(z_mean) - backend.exp(z_log_var))
vae_loss            = backend.mean(reconstruction_loss + kl_loss)
# Add metric: print out each loss during every epoch
vae.add_loss(reconstruction_loss)
vae.add_loss(kl_loss)
vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
vae.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')

# Compile
vae.compile(optimizer=optimizers.Adam())

# Train
epochs, batch_size = 10000, 64
# Convert x_train and x_val to a TensorFlow dataset
train_dataset      = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset      = train_dataset.batch(batch_size)
train_dataset      = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)   # prefetch the data to overlap the preprocessing and training steps

val_dataset        = tf.data.Dataset.from_tensor_slices(x_val)
val_dataset        = val_dataset.batch(batch_size)

# Fit model
history = vae.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[callback])
print(len(history.history['loss']))

# Save Model
vae.save(f'../../TemporaryData/VAE_model/vae_s{train_size}_e{epochs}_b{batch_size}.h5', save_format="tf")

# Save loss log
import pickle
with open(f'../../TemporaryData/VAE_model/vae_s{train_size}_e{epochs}_b{batch_size}_loss_log.txt', 'wb') as file_txt:
  pickle.dump(history.history, file_txt)

# Save training, validation, test size dataset
save_dataset = {'x_train':x_train, 'x_val':x_val, 'x_test':x_test}
np.save(f'../../TemporaryData/VAE_model/dataset_file/vae_s{train_size}_e{epochs}_b{batch_size}.npy', save_dataset) 

           
