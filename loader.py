import tensorflow as tf

BUFFER_SIZE = 60000
BATCH_SIZE = 256

def load(batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE):
    
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    
    train_images = (train_images - 127.5) / 127.5

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size,
                                                                                                True)
    return train_dataset

