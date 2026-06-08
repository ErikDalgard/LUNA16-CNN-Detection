# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# %%
model_1_input = (20, 20, 6, 1)
model_2_input = (30, 30, 10, 1)
model_3_input = (40, 40, 26, 1)

# %%
#ARCHI-1
#-------

def get_archi_1_3D(input_shape=(model_1_input)):   
    archi_1_3D = keras.Sequential([
        #Input layer, input (20x20x6)
        layers.Input(shape=model_1_input),

        #BLOCK 1
        layers.Conv3D(64, kernel_size=(5, 5, 3), activation='relu', padding='same'),
        #1st Pooling, reducing 20x20x6 to 10x10x3
        layers.MaxPooling3D(pool_size=2),

        #BLOCK 2
        layers.Conv3D(64, kernel_size=(1, 1, 1), activation='relu', padding='same'),

        #BLOCK 3
        layers.Conv3D(64, kernel_size=(5, 5, 3), activation='relu', padding='same'),
        #2nd Pooling, reducing 10x10x3 to 5x5x1
        layers.MaxPooling3D(pool_size=2),

        #BLOCK 4
        layers.Conv3D(64, kernel_size=(5, 5, 1), activation='relu', padding='same'),
        
        #Classification block
        layers.GlobalAveragePooling3D(),
        layers.Dense(150, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ], name="archi_1_3D")
    return archi_1_3D

# %%
#ARCHI-2
#-------

def get_archi_2_3D(input_shape=(model_2_input)):

    archi_2_3D = keras.Sequential([
        #Input layer, input (30x30x10)
        layers.Input(shape=model_2_input),

        #BLOCK 1
        layers.Conv3D(64, kernel_size=(5, 5, 3), activation='relu', padding='same'),
        #1st Pooling, reducing 30x30x10 to 15x15x5
        layers.MaxPooling3D(pool_size=2), 

        #BLOCK 2
        layers.Conv3D(64, kernel_size=(2, 2, 1), activation='relu', padding='same'),

        #BLOCK 3
        layers.Conv3D(64, kernel_size=(5, 5, 3), activation='relu', padding='same'),
        #2nd Pooling, reducing 15x15x5 to 7x7x2
        layers.MaxPooling3D(pool_size=2),

        #BLOCK 4
        layers.Conv3D(64, kernel_size=(5, 5, 3), activation='relu', padding='same'),

        #Classification block
        layers.GlobalAveragePooling3D(),
        layers.Dense(250, activation='relu'),
        layers.Dense(1, activation='sigmoid')
        ], name="archi_2_3D")

    return archi_2_3D

# %%
#ARCHI-3
#-------

def get_archi_3_3D(input_shape=(model_3_input)):
    archi_3_3D = keras.Sequential([
        #Input layer, input (40x40x26)
        layers.Input(shape=model_3_input),

        #BLOCK 1
        layers.Conv3D(64, kernel_size=(5, 5, 3), activation='relu', padding='same'),
        #1st Pooling, reducing 40x40x26 to 20x20x13
        layers.MaxPooling3D(pool_size=2), 

        #BLOCK 2
        layers.Conv3D(64, kernel_size=(2, 2, 2), activation='relu', padding='same'),

        #BLOCK 3
        layers.Conv3D(64, kernel_size=(5, 5, 3), activation='relu', padding='same'),
        #2nd Pooling, reducing 20x20x13 to 10x10x6
        layers.MaxPooling3D(pool_size=2), 

        #BLOCK 4
        layers.Conv3D(64, kernel_size=(5, 5, 3), activation='relu', padding='same'),

        #Classification block
        layers.GlobalAveragePooling3D(),
        layers.Dense(250, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ], name="archi_3_3D")
    return archi_3_3D
