# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# %%
model_1_input = (20, 20, 6)
model_2_input = (30, 30, 10)
model_3_input = (40, 40, 26)

# %%
#ARCHI-1
#-------

def get_archi_1_AE(input_shape=model_1_input):
    archi_1_AE = keras.Sequential([
        #Input layer, input (20x20x6)
        layers.Input(shape=input_shape),

        #BLOCK 1 (Encoder)
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        #1st Pooling, reducing 20x20 to 10x10
        layers.MaxPooling2D(pool_size=2),

        #BLOCK 2 (Encoder)
        layers.Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same'),

        #BLOCK 3 (Encoder)
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        #2nd Pooling, reducing 10x10 to 5x5 (Bottleneck)
        layers.MaxPooling2D(pool_size=2),

        #BLOCK 4 (Encoder Center)
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),

        #BLOCK 5 (Decoder Center Mirror)
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),

        #BLOCK 6 (Decoder)
        #1st Upsampling, expanding 5x5 to 10x10
        layers.UpSampling2D(size=2),
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        layers.Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same'),

        #BLOCK 7 (Decoder)
        #2nd Upsampling, expanding 10x10 to 20x20
        layers.UpSampling2D(size=2),

        #Reconstruction Output Layer
        #Matches original 6 channels. Sigmoid maps pixel intensities between 0 and 1.
        layers.Conv2D(6, kernel_size=(5, 5), activation='sigmoid', padding='same')
    ], name="archi_1_AE")
    return archi_1_AE

# %%
#ARCHI-2
#-------

def get_archi_2_AE(input_shape=model_2_input):
    archi_2_AE = keras.Sequential([
        #Input layer, input (30x30x10)
        layers.Input(shape=input_shape),

        #BLOCK 1 (Encoder)
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        #1st Pooling, reducing 30x30 to 15x15
        layers.MaxPooling2D(pool_size=2), 

        #BLOCK 2 (Encoder)
        layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),

        #BLOCK 3 (Encoder)
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        #2nd Pooling, reducing 15x15 to 7x7 (Bottleneck)
        layers.MaxPooling2D(pool_size=2),

        #BLOCK 4 (Encoder Center)
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),

        #BLOCK 5 (Decoder Center Mirror)
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),

        #BLOCK 6 (Decoder)
        #1st Upsampling, expanding 7x7 to 14x14
        layers.UpSampling2D(size=2),
        # Fix odd-number rounding difference from pooling (15 -> 7 -> 14) by padding 1 pixel
        layers.ZeroPadding2D(padding=((0, 1), (0, 1))), # Pads up to 15x15
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),

        #BLOCK 7 (Decoder)
        #2nd Upsampling, expanding 15x15 to 30x30
        layers.UpSampling2D(size=2),

        #Reconstruction Output Layer (Matches original 10 channels)
        layers.Conv2D(10, kernel_size=(5, 5), activation='sigmoid', padding='same')
    ], name="archi_2_AE")
    return archi_2_AE

# %%
#ARCHI-3
#-------

def get_archi_3_AE(input_shape=model_3_input):
    archi_3_AE = keras.Sequential([
        #Input layer, input (40x40x26)
        layers.Input(shape=input_shape),

        #BLOCK 1 (Encoder)
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        #1st Pooling, reducing 40x40 to 20x20
        layers.MaxPooling2D(pool_size=2), 

        #BLOCK 2 (Encoder)
        layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),

        #BLOCK 3 (Encoder)
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        #2nd Pooling, reducing 20x20 to 10x10 (Bottleneck)
        layers.MaxPooling2D(pool_size=2), 

        #BLOCK 4 (Encoder Center)
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),

        #BLOCK 5 (Decoder Center Mirror)
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),

        #BLOCK 6 (Decoder)
        #1st Upsampling, expanding 10x10 to 20x20
        layers.UpSampling2D(size=2), 
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),

        #BLOCK 7 (Decoder)
        #2nd Upsampling, expanding 20x20 to 40x40
        layers.UpSampling2D(size=2),

        #Reconstruction Output Layer (Matches original 26 channels)
        layers.Conv2D(26, kernel_size=(5, 5), activation='sigmoid', padding='same')
    ], name="archi_3_AE")
    return archi_3_AE