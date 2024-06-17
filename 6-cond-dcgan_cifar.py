import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import ops

import tensorflow as tf

import numpy as np
import datetime

import utils as U
import gan_models as M

# CONSTANTS
EPOCHS = 100
BATCH_SIZE = 128
N_SAMPLES = 64
MODEL_DIR = "/Users/mghifary/Work/Code/AI/models"
NC = 3 # number of channels
NZ = 128 # latent dimension
NGF = 64 # number of generator filters
NDF = 64 # number of discriminator filters
IMAGE_DIM = 32 # image dimension

class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, latent_variable=None, condition_variable=None):
        super().__init__()

        self.train_steps = 0
        if latent_variable is not None:
            self.latent_variable = latent_variable
        else:
            self.latent_variable = np.random.normal(size=(N_SAMPLES, NZ))

        if condition_variable is not None:
            self.condition_variable = condition_variable
        else:
            self.condition_variable = np.random.randint(0, 10, N_SAMPLES)

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        if self.train_steps % 100 == 0:
            # print(f" ... [{self.train_steps}] got log keys: {keys}")
            fake_images = model.cond_generator([self.latent_variable, self.condition_variable])
            fake_samples = np.reshape(fake_images, (-1, IMAGE_DIM, IMAGE_DIM, NC))

            checkpoint_path = os.path.join(checkpoint_dir, f"fake_images_step-{self.train_steps}.jpg")
            print(f" ... [{self.train_steps}] Visualize {checkpoint_path}")
            U.visualize_grid(
                fake_samples, 
                figpath=checkpoint_path
            )

        self.train_steps += 1


# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize data to (-1, 1)
x_train = U.normalize_batch(X_train, low_s=0, high_s=255, low_t=-1, high_t=1)
x_test = U.normalize_batch(X_test, low_s=0, high_s=255, low_t=-1, high_t=1)

x_train = ops.image.resize(x_train, (IMAGE_DIM, IMAGE_DIM))
x_test = ops.image.resize(x_test, (IMAGE_DIM, IMAGE_DIM))

y_train = y_train.flatten()
y_test = y_test.flatten()

# Create tf data
all_images = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])
num_classes = len(np.unique(all_labels))

dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)


# disc_net = M.create_cond_discriminator(input_shape=(784, ), condition_size=num_classes)
input_shape = (IMAGE_DIM, IMAGE_DIM, NC)    
disc_net = M.create_cond_dcdiscriminator(input_shape, NDF, num_classes)
# print(disc_net.summary())

gen_net = M.create_cond_dcgenerator(
    NZ,
    NGF,
    NC,
    num_classes
)
# print(gen_net.summary())

model = M.CondDCGAN(
    discriminator=disc_net, 
    generator=gen_net, 
    latent_dim=NZ,
    condition_size=num_classes
)
print(model.summary(expand_nested=True))

model.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
)

# Define callback for Model Checkpoint
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint_dir = os.path.join(MODEL_DIR, f"cond-dcgan-cifar10-keras-{current_time}")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

samples = all_images[:N_SAMPLES]

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

# cond_y = ops.one_hot(y_test, num_classes=num_classes)

# ## for discriminator
# cond_image_y = ops.expand_dims(cond_y, axis=(1, 2))
# cond_imagerep_y = ops.repeat(cond_image_y, x_test[0].shape[0], axis=1)
# cond_imagerep_y = ops.repeat(cond_imagerep_y, x_test[0].shape[1], axis=2)


z = np.random.normal(size=(N_SAMPLES, NZ)) # fixed latent variable for visualization

row = int(np.round(np.sqrt(N_SAMPLES)))
cond_y = np.repeat(np.arange(row), row)
cond_y = keras.utils.to_categorical(cond_y, num_classes=num_classes)

display_callback = DisplayCallback(
    latent_variable=z,
    condition_variable=cond_y
)

model.fit(
    dataset, 
    epochs=EPOCHS,
    callbacks=[modelcp_callback, display_callback]
)