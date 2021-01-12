import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from net import *
from  loader import *

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time

from IPython import display

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
image_dir = 'images/wgan_local'
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

EPOCHS=50
noise_dim = 100
num_of_examples = 16
n_critic = 4
grad_penalty = 50

seed = tf.random.normal([num_of_examples, noise_dim])

def discriminator_loss(real_predictions, fake_predictions):
    real_loss = cross_entropy(tf.ones_like(real_predictions), real_predictions)
    fake_loss = cross_entropy(tf.zeros_like(fake_predictions), fake_predictions)

    return real_loss + fake_loss

def generator_loss(fake_predictions):
    return cross_entropy(tf.ones_like(fake_predictions), fake_predictions)

def wg_loss(data, G, D):
    batch_size = data.shape[0]
    noise = tf.random.normal( [batch_size, noise_dim] )
    return -tf.math.reduce_mean ( D(G(noise)) )

# Computed per critic training step
def wd_loss(data, G, D):
    batch_size = data.shape[0]
    z = tf.random.normal( [batch_size, noise_dim] )
    xtilde = G(z)
    assert data.shape == xtilde.shape

    eps = tf.random.uniform( [batch_size, 1, 1, 1] )

    #print( "Data shape %s " %(data.shape, ))
    xhat = tf.Variable(data * eps + xtilde * (1-eps))


    with tf.GradientTape() as tape:
        l = D(xhat)

    assert l.shape == [batch_size, 1]
    gradnorm = tf.norm( tf.reshape( tape.gradient(l, xhat), [batch_size, -1] ), axis=-1 )

    gradloss = tf.math.reduce_mean((gradnorm - 1)**4)
    wloss = tf.math.reduce_mean( D(xtilde) - D(data) )
    print("Mean grad: %.2f, grad loss: %.2f, wloss: %.2f" % (tf.math.reduce_mean(gradnorm),
                                                             gradloss, wloss))
    
    return wloss + grad_penalty *  gradloss
    
    
    
    

    

def train_step(images, G, D, G_optimizer, D_optimizer):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = G(noise)
        real_predictions = D(images)
        fake_predictions = D(generated_images)
        g_loss = generator_loss(fake_predictions)
        d_loss = discriminator_loss(real_predictions, fake_predictions)

    gen_gradients = gen_tape.gradient(g_loss, G.trainable_variables)
    disc_gradients = disc_tape.gradient(d_loss, D.trainable_variables)
    G_optimizer.apply_gradients(zip(gen_gradients, G.trainable_variables))
    D_optimizer.apply_gradients(zip(disc_gradients, D.trainable_variables))

    return g_loss, d_loss

def wgan_train_step(images, G, D, G_optimizer, D_optimizer):
    subbatch_size = int(images.shape[0] / n_critic)
    # print("Subbatch size %d " % subbatch_size)
    d_loss = 0.0
    for j in range(0, n_critic):
        subbatch = images[j*subbatch_size:(j+1)*subbatch_size, :]
        with tf.GradientTape() as disc_tape:
            d_loss = wd_loss(subbatch, G, D)
        D_gradients = disc_tape.gradient(d_loss, D.trainable_variables)
        D_optimizer.apply_gradients(zip(D_gradients, D.trainable_variables))

    with tf.GradientTape() as gen_tape:
        g_loss = wg_loss(images, G, D)
    G_gradients = gen_tape.gradient(g_loss, G.trainable_variables)
    G_optimizer.apply_gradients(zip(G_gradients, G.trainable_variables))

    return g_loss, d_loss

def train(dataset, epochs, G, D, G_optimizer, D_optimizer, checkpoint):
    for e in range(epochs):
        start = time.time()
        for i, batch in enumerate(dataset):
            # TODO: add timing

            g_loss, d_loss = wgan_train_step(batch, G, D, G_optimizer, D_optimizer)

            if i % 10 == 9:
                print("Epoch %d, Batch %d: Generator loss %.2f, Discriminator loss %.2f" %
                      (e+1, i+1, g_loss.numpy(), d_loss.numpy()))
                

        print("Running time for epoch %d: %.2f seconds" % (e+1, time.time() - start))

        display.clear_output(wait=True)
        generate_and_save_images(G, e, seed)
        if e % 10 == 9:
            checkpoint.save(file_prefix = checkpoint_prefix)


def generate_and_save_images(generator, epoch, test_input):
    fig = plt.figure(figsize=(4,4))

    sample = generator(test_input)

    for i in range(sample.shape[0]):
        plt.subplot(4, 4, i+1)
        # Rescale to [0, 255]
        plt.imshow(sample[i, :, :, 0]*127.5 + 127.5, cmap='gray')

        plt.axis('off')
    plt.savefig(image_dir + '/' + 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    
if __name__ == '__main__':
    GPUs = tf.config.experimental.list_physical_devices("GPU")
    CPUs = tf.config.experimental.list_physical_devices("CPU")
    print("Num CPUs: %d, Num GPUs: %d" % (len(CPUs), len(GPUs)))
    
    generator_optimizer = keras.optimizers.Adam(0.0001)
    discriminator_optimizer = keras.optimizers.Adam(0.0001)

    generator = make_generator_model()
    discriminator = make_discriminator_model()
    

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimzier=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    
    


    dataset = load()

    train(dataset, EPOCHS, generator, discriminator,
          generator_optimizer, discriminator_optimizer,
          checkpoint)
    
