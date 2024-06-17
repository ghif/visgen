"""
The code provided is a Python script that trains a diffusion model for image denoising using the CIFAR-10 dataset. The diffusion model is implemented using the Keras library with TensorFlow as the backend. The script imports necessary libraries and defines various constants and parameters for the model. It also includes custom classes and functions for visualization, evaluation metrics, and model architecture. The script loads the CIFAR-10 dataset, preprocesses the data, and creates TensorFlow datasets for training and validation. The diffusion model is then compiled and trained using the training dataset. During training, the model periodically generates denoised images and saves them as checkpoints. The best model based on the validation KID metric is also saved.
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import ops

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import datetime

import utils as U
import gan_models as M

import math
import matplotlib.pyplot as plt

MODEL_DIR = "/Users/mghifary/Work/Code/AI/models"

# Data
image_size = 32

# KID = Kernel Inception Distance
kid_image_size  = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20

# Sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# Architecture
embedding_dims = 32
embedding_min_frequency = 1.0
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

# Optimization
num_epochs = 50
batch_size = 128
num_samples = 64
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4

class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, latent_variable=None):
        super().__init__()
        self.train_steps = 0
        if latent_variable is not None:
            self.latent_variable = latent_variable
        else:
            self.latent_variable = keras.random.normal(
                shape=(num_samples, image_size, image_size, 3)
            )

    def on_train_batch_end(self, batch, logs=None):
        if self.train_steps % 100 == 0:
        # if self.train_steps % 2 == 0:
            # print(f" ... [{self.train_steps}] got log keys: {keys}")
            generated_images = model.generate(
                self.latent_variable,
                plot_diffusion_steps
            )
            checkpoint_path = os.path.join(checkpoint_dir, f"denoised_images_step-{self.train_steps}.jpg")
            print(f" ... [{self.train_steps}] Visualize {checkpoint_path}")
            U.visualize_grid(
                generated_images, 
                figpath=checkpoint_path
            )

        self.train_steps += 1

@keras.saving.register_keras_serializable()
class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(image_size, image_size, 3)),
                keras.layers.Rescaling(255.0),
                keras.layers.Resizing(height=kid_image_size, width=kid_image_size),
                keras.layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                keras.layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = ops.cast(ops.shape(features_1)[1], dtype="float32")
        return (
            features_1 @ ops.transpose(features_2) / feature_dimensions + 1.0
        ) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        """
        Update the KID metric with real and generated images

        Args:
            real_images (tf.Tensor): the real images
            generated_images (tf.Tensor): the generated images
            sample_weight (): (optional) ignored
        """
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = generated_images.shape[0]
        # batch_size = ops.shape(real_features)[0].numpy()
        print(f"[update_state] real_images.shape: {real_images.shape}")
        print(f"[update_state] generated_images.shape: {generated_images.shape}")
        print(f"[update_state] real_features.shape: {real_features.shape}")
        print(f"[update_state] batch_size: {batch_size}")
        print(f"[update_state] kernel_real.shape: {kernel_real.shape}")
        batch_size_f = ops.cast(batch_size, dtype="float32")
        mean_kernel_real = ops.sum(kernel_real * (1.0 - ops.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = ops.sum(
            kernel_generated * (1.0 - ops.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = ops.mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()

@keras.saving.register_keras_serializable()
def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = ops.exp(
        ops.linspace(
            ops.log(embedding_min_frequency),
            ops.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )

    angular_speeds = ops.cast(2.0 * math.pi * frequencies, dtype="float32")
    embeddings = ops.concatenate(
        [ops.sin(angular_speeds * x), ops.cos(angular_speeds * x)], axis=3
    )
    return embeddings

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = keras.layers.Conv2D(width, kernel_size=1)(x)
        
        x = keras.layers.BatchNormalization(center=False, scale=False)(x)
        x = keras.layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x)
        x = keras.layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = keras.layers.Add()([x, residual])
        return x
    
    return apply

def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = keras.layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply

def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = keras.layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply

def create_unet(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = keras.layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, 32))(noise_variances)
    e = keras.layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = keras.layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = keras.layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])
    
    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)
    
    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])
    
    x = keras.layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")

@keras.saving.register_keras_serializable()
class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()
        self.normalizer = keras.layers.Normalization()
        self.network = create_unet(image_size, widths, block_depth)
        self.ema_network = create_unet(image_size, widths, block_depth)

    def compile(self, **kwargs):
        super().compile(**kwargs)
        # self.noise_loss_tracker = keras.metrics.Metric(name="n_loss")
        # self.image_loss_tracker = keras.metrics.Metric(name="i_loss")
        self.noise_loss_tracker = keras.metrics.MeanAbsoluteError(name="n_loss")
        self.image_loss_tracker = keras.metrics.MeanAbsoluteError(name="i_loss")
        self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]
    
    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        return ops.clip(images, 0.0, 1.0)
    
    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = ops.cast(ops.arccos(max_signal_rate), "float32")
        end_angle = ops.cast(ops.arccos(min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = ops.cos(diffusion_angles)
        noise_rates = ops.sin(diffusion_angles) 

        return noise_rates, signal_rates
    
    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images
    
    def reverse_diffusion(self, initial_noise, diffusion_steps):
        """
        Reverse the diffusion process to generate images from noise
        
        Args:
            initial_noise: the initial noise tensor
            diffusion_steps: the number of diffusion steps to reverse

        Returns:
            pred_images: the predicted images
        """

        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # Important: at the first sampling step, the "noisy image" is purse noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images
    
    def generate(self, initial_noise, diffusion_steps):
        """
        Generate images from noise

        Args:
            initial_noise (tf.Tensor): the initial noise tensor
            diffusion_steps (int): the number of diffusion steps to reverse
        
        Returns:
            generated_images (tf.Tensor): the generated images
        """
        # noise -> images -> denormalized images
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images
    
    def train_step(self, images):
        """
        Train the diffusion model on a batch of images

        Args:
            images (tf.Tensor): the batch of images
        
        Returns:
            metrics (dict): the metrics to log
        """
        # normalize images to have standard deviation of 1, like the noises
        batch_size = ops.shape(images)[0]
        images = self.normalizer(images, training=True)
        noises = keras.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )

        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises) # used for training
            # image_loss = self.loss(images, pred_images) # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noises, pred_noises)
        self.image_loss_tracker.update_state(images, pred_images)

        # track the exponential moving average of the network weights
        for weight, ema_weights in zip(self.network.weights, self.ema_network.weights):
            ema_weights.assign(ema * ema_weights + (1-ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}
    
    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        batch_size_ops = ops.shape(images)[0]
        images = self.normalizer(images, training=False)
        noises = keras.random.normal(shape=(batch_size_ops, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch_size_ops, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to seperate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        self.image_loss_tracker.update_state(images, pred_images)
        self.noise_loss_tracker.update_state(noises, pred_noises)

        # # measure KID between real and generated images
        # # this is computationally demanding, kid_diffusion_steps has to be small
        # images = self.denormalize(images)
        # generated_images = self.generate(
        #     batch_size, 
        #     kid_diffusion_steps
        # )

        # print(f"images shape: {images.shape}")
        # print(f"type(images): {type(images)}")  
        # print(f"generated_images shape: {generated_images.shape}")
        # print(f"type(generated_images: {type(generated_images)}")
        # self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# # Normalize data to (-1, 1)
x_train = U.normalize_batch(X_train, low_s=0, high_s=255, low_t=0, high_t=1)
x_test = U.normalize_batch(X_test, low_s=0, high_s=255, low_t=0, high_t=1)

x_train = ops.image.resize(x_train, (image_size, image_size))
x_test = ops.image.resize(x_test, (image_size, image_size))

# Create tf data
# all_images = np.concatenate([x_train, x_test])
dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices(x_test)
val_dataset = val_dataset.batch(batch_size)

# test resblock
net = create_unet(image_size, widths, block_depth)

model = DiffusionModel(image_size, widths, block_depth)
model.summary(expand_nested=True)

loss_fn = keras.losses.MeanAbsoluteError()
model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=loss_fn,
)

# save the best model based on the validation KID metric
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = os.path.join(MODEL_DIR, f"ddim-cifar10-keras-{current_time}")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_path = os.path.join(checkpoint_dir, "model.keras")
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    # save_weights_only=True,
    monitor="val_kid",
    mode="min",
    save_best_only=True,
)

display_callback = DisplayCallback()

# calculate mean and variance of training dataset for normalization
model.normalizer.adapt(dataset)

# run training and plot generated images periodically
model.fit(
    dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    callbacks=[
        display_callback, 
        checkpoint_callback,
    ],
)