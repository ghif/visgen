import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

import tensorflow as tf

import numpy as np
import datetime

import utils as U
import gan_models as M

# CONSTANTS
EPOCHS = 50
LATENT_DIM = 100
BATCH_SIZE = 128
N_SAMPLES = 64
MODEL_DIR = "/Users/mghifary/Work/Code/AI/models"

class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, latent_variable=None):
        super().__init__()

        self.train_steps = 0
        if latent_variable is not None:
            self.latent_variable = latent_variable
        else:
            self.latent_variable = np.random.normal(size=(N_SAMPLES, LATENT_DIM))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        if self.train_steps % 100 == 0:
            # print(f" ... [{self.train_steps}] got log keys: {keys}")
            fake_images = model.generator(self.latent_variable)
            fake_samples = np.reshape(fake_images, (-1, 28, 28, 1))

            checkpoint_path = os.path.join(checkpoint_dir, f"fake_images_step-{self.train_steps}.jpg")
            print(f" ... [{self.train_steps}] Visualize {checkpoint_path}")
            U.visualize_grid(
                fake_samples, 
                figpath=checkpoint_path
            )

        self.train_steps += 1


# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(X_train, (-1, 784)).astype(np.float32)
x_test = np.reshape(X_test, (-1, 784)).astype(np.float32)

# Normalize data to (-1, 1)
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

# Create tf data
all_images = np.concatenate([x_train, x_test])
dataset = tf.data.Dataset.from_tensor_slices(all_images)
dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)


disc_net = M.create_discriminator(input_shape=(784, ))
# print(disc_net.summary())

gen_net = M.create_generator(latent_size=LATENT_DIM, output_size=784)
# print(gen_net.summary())

model = M.GAN(
    discriminator=disc_net, 
    generator=gen_net, 
    latent_dim=LATENT_DIM
)
print(model.summary(expand_nested=True))

model.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
)

# Define callback for Model Checkpoint
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint_dir = os.path.join(MODEL_DIR, f"gan-mnist-keras-{current_time}")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

samples = np.reshape(all_images[:N_SAMPLES], (-1, 28, 28, 1))

# Show real images
U.visualize_grid(
    samples, 
    figpath=os.path.join(checkpoint_dir, "real_images.jpg")
)

checkpoint_filepath = os.path.join(checkpoint_dir, "model.keras")

modelcp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    # save_weights_only=True
)

z = np.random.normal(size=(N_SAMPLES, LATENT_DIM)) # fixed latent variable for visualization
display_callback = DisplayCallback(
    latent_variable=z
)

model.fit(
    dataset, 
    epochs=EPOCHS,
    callbacks=[modelcp_callback, display_callback]
)