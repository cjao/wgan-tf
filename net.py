import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_generator_model():
    model = keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.ReLU())
    model.add(layers.BatchNormalization())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, 3, strides=(1, 1), padding='same'))
    model.add(layers.ReLU())
    model.add(layers.BatchNormalization())

    assert model.output_shape == (None, 7, 7, 128)

    model.add(layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same'))
    model.add(layers.ReLU())
    model.add(layers.BatchNormalization())

    assert model.output_shape == (None, 14, 14, 64)
    
    model.add(layers.Conv2DTranspose(1, 3, strides=(2, 2), padding='same'))
    model.add(layers.Activation('tanh'))

    assert model.output_shape == (None, 28, 28, 1)

    return model
    

    
def make_discriminator_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, 3, strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU(0.2))
    assert model.output_shape == (None, 14, 14, 64)

    model.add(layers.Conv2D(128, 3, strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(0.2))
    assert model.output_shape == (None, 7, 7, 128)

    model.add(layers.Flatten())

    model.add(layers.Dense(1))

    return model
           

# For W-GAN
def make_critic_model():
    return make_discriminator_model()
    
