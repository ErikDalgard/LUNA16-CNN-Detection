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

def get_archi_1_2D(input_shape=(model_1_input)):    
    archi_1_2D = keras.Sequential([
        #Input layer, input (20x20x6)
        layers.Input(shape=model_1_input),

        #BLOCK 1
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        #1st Pooling, reducing 20x20 to 10x10
        layers.MaxPooling2D(pool_size=2),

        #BLOCK 2
        layers.Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same'),

        #BLOCK 3
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        #2nd Pooling, reducing 10x10 to 5x5
        layers.MaxPooling2D(pool_size=2),

        #BLOCK 4
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),

        #Classification block
        layers.Flatten(),
        layers.Dense(150, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ], name="2D_archi_1")
    return archi_1_2D
# %%
#ARCHI-2
#-------

def get_archi_2_2D(input_shape=(model_2_input)):
    archi_2_2D = keras.Sequential([
        #Input layer, input (30x30x10)
        layers.Input(shape=model_2_input),

        #BLOCK 1
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        #1st Pooling, reducing 30x30 to 15x15
        layers.MaxPooling2D(pool_size=2), 

        #BLOCK 2
        layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),

        #BLOCK 3
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        #2nd Pooling, reducing 15x15 to 7x7
        layers.MaxPooling2D(pool_size=2),

        #BLOCK 4
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),

        #Classification block
        layers.Flatten(),
        layers.Dense(250, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ], name="2D_archi_2")
    return archi_2_2D


# %%
#ARCHI-3
#-------

def get_archi_3_2D(input_shape=(model_3_input)):
    archi_3_2D = keras.Sequential([
    #Input layer, input (40x40x26)
    layers.Input(shape=model_3_input),

    #BLOCK 1
    layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
    #1st Pooling, reducing 40x40 to 20x20
    layers.MaxPooling2D(pool_size=2), 

    #BLOCK 2
    layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),

    #BLOCK 3
    layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
    #2nd Pooling, reducing 20x20 to 10x10
    layers.MaxPooling2D(pool_size=2), 

    #BLOCK 4
    layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),

    #Classification block
    layers.Flatten(),
    layers.Dense(250, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
    ], name="2D_archi_3")
    return archi_3_2D



