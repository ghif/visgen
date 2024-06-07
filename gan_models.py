import keras
from keras import activations
from keras import ops

import tensorflow as tf

class CondGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim=100, condition_size=10):
        super(CondGAN, self).__init__()
        self.cond_discriminator = discriminator
        self.cond_generator = generator
        self.latent_dim = latent_dim
        self.condition_size = condition_size

        self.seed_generator = keras.random.SeedGenerator(1337) # for reproducibility
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]
    
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, batch_data):
        ############################
        #  Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        
        # real images
        real_x, real_label = batch_data
        cond_y = ops.one_hot(real_label, num_classes=self.condition_size)

        batch_size = ops.shape(real_x)[0]

        real_y = ops.ones((batch_size, 1))

        # fake images

        ## latent variable
        z = keras.random.normal(
            shape=(batch_size, self.latent_dim),
            seed=self.seed_generator
        )

        fake_x = self.cond_generator([z, cond_y])
        fake_y = ops.zeros((batch_size, 1))


        # Assemble training data
        combined_images = ops.concatenate(
            [real_x, fake_x], axis=0
        )
        combined_cond_y = ops.concatenate(
            [cond_y, cond_y], axis=0
        )

        combined_labels = ops.concatenate(
            [real_y, fake_y], axis=0
        )

        # Train the discriminator
        with tf.GradientTape() as tape:
            combined_preds = self.cond_discriminator([combined_images, combined_cond_y])
            d_loss = self.loss_fn(combined_labels, combined_preds)

        grads = tape.gradient(d_loss, self.cond_discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.cond_discriminator.trainable_weights)
        )

        ############################
        # Update G network: minimize log(D(x)) + log(1 - D(G(z)))
        ###########################
        z = keras.random.normal(
            shape=(batch_size, self.latent_dim),
            seed=self.seed_generator
        )
        
        # Train the generator
        with tf.GradientTape() as tape:
            fake_x = self.cond_generator([z, cond_y])
            fake_misleading_y = ops.ones((batch_size, 1))
            preds = self.cond_discriminator([fake_x, cond_y])
            g_loss = self.loss_fn(fake_misleading_y, preds)
        
        grads = tape.gradient(g_loss, self.cond_generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.cond_generator.trainable_weights)
        )

        # Monitor loss
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result()
        }


class CondDCGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim=100, condition_size=10):
        super(CondDCGAN, self).__init__()
        self.cond_discriminator = discriminator
        self.cond_generator = generator
        self.latent_dim = latent_dim
        self.condition_size = condition_size

        self.seed_generator = keras.random.SeedGenerator(1337) # for reproducibility
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]
    
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, batch_data):
        ############################
        #  Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        
        # real images
        real_x, real_label = batch_data

        # target conditioning
        ## for generator
        cond_y = ops.one_hot(real_label, num_classes=self.condition_size)

        ## for discriminator
        # cond_image_y = ops.expand_dims(cond_y, axis=(1, 2))
        # cond_imagerep_y = ops.repeat(cond_image_y, real_x[0].shape[0], axis=1)
        # cond_imagerep_y = ops.repeat(cond_imagerep_y, real_x[0].shape[1], axis=2)
        cond_image_y = cond_y[:, :, None, None]
        cond_imagerep_y = ops.repeat(cond_image_y, repeats=[real_x[0].shape[0] * real_x[0].shape[1]])
        cond_imagerep_y = ops.reshape(cond_imagerep_y, (-1, real_x[0].shape[0], real_x[0].shape[1], self.condition_size))

        batch_size = ops.shape(real_x)[0]

        real_y = ops.ones((batch_size, 1))

        # fake images

        ## latent variable
        z = keras.random.normal(
            shape=(batch_size, self.latent_dim),
            seed=self.seed_generator
        )

        fake_x = self.cond_generator([z, cond_y])
        fake_y = ops.zeros((batch_size, 1))


        # Assemble training data
        real_x_cond = ops.concatenate([real_x, cond_imagerep_y], axis=-1)
        fake_x_cond = ops.concatenate([fake_x, cond_imagerep_y], axis=-1)

        combined_inputs = ops.concatenate(
            [real_x_cond, fake_x_cond], axis=0
        )
        
        combined_labels = ops.concatenate(
            [real_y, fake_y], axis=0
        )

        # Train the discriminator
        with tf.GradientTape() as tape:
            combined_preds = self.cond_discriminator(combined_inputs)
            d_loss = self.loss_fn(combined_labels, combined_preds)

        grads = tape.gradient(d_loss, self.cond_discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.cond_discriminator.trainable_weights)
        )

        ############################
        # Update G network: minimize log(D(x)) + log(1 - D(G(z)))
        ###########################
        z = keras.random.normal(
            shape=(batch_size, self.latent_dim),
            seed=self.seed_generator
        )
        
        # Train the generator
        with tf.GradientTape() as tape:
            fake_x = self.cond_generator([z, cond_y])
            fake_x_cond = ops.concatenate([fake_x, cond_imagerep_y], axis=-1)
            fake_misleading_y = ops.ones((batch_size, 1))
            preds = self.cond_discriminator(fake_x_cond)
            g_loss = self.loss_fn(fake_misleading_y, preds)
        
        grads = tape.gradient(g_loss, self.cond_generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.cond_generator.trainable_weights)
        )

        # Monitor loss
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result()
        }


    
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim=100):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

        self.seed_generator = keras.random.SeedGenerator(1337) # for reproducibility
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]
    
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, batch_data):
        ############################
        #  Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        
        # real images
        real_x = batch_data
        batch_size = ops.shape(real_x)[0]

        real_y = ops.ones((batch_size, 1))

        # fake images
        z = keras.random.normal(
            shape=(batch_size, self.latent_dim),
            seed=self.seed_generator
        )
        fake_x = self.generator(z)
        fake_y = ops.zeros((batch_size, 1))


        # Assemble training data
        combined_images = ops.concatenate(
            [real_x, fake_x], axis=0
        )
        combined_labels = ops.concatenate(
            [real_y, fake_y], axis=0
        )

        # Train the discriminator
        with tf.GradientTape() as tape:
            combined_preds = self.discriminator(combined_images)
            d_loss = self.loss_fn(combined_labels, combined_preds)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        ############################
        # Update G network: minimize log(D(x)) + log(1 - D(G(z)))
        ###########################
        z = keras.random.normal(
            shape=(batch_size, self.latent_dim),
            seed=self.seed_generator
        )
        
        # Train the generator
        with tf.GradientTape() as tape:
            fake_x = self.generator(z)
            fake_misleading_y = ops.ones((batch_size, 1))
            preds = self.discriminator(fake_x)
            g_loss = self.loss_fn(fake_misleading_y, preds)
        
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )

        # Monitor loss
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result()
        }

def create_discriminator(input_shape, num_classses=1):
    inputs = keras.Input(shape=input_shape)
    h = keras.layers.Dense(512)(inputs)
    h = activations.leaky_relu(h, negative_slope=0.2)
    h = keras.layers.Dense(256)(h)
    h = activations.leaky_relu(h, negative_slope=0.2)
    y = keras.layers.Dense(num_classses)(h)
    return keras.Model(inputs=inputs, outputs=y, name="discriminator")

def create_generator(latent_size=100, output_size=784):
    # latent (100) -> linear (128)
    inputs = keras.Input(shape=(latent_size, ))

    h = keras.layers.Dense(128)(inputs)
    h = activations.leaky_relu(h, negative_slope=0.2)

    h = keras.layers.Dense(256)(h)
    h = keras.layers.BatchNormalization()(h)
    h = activations.leaky_relu(h, negative_slope=0.2)

    h = keras.layers.Dense(512)(h)
    h = keras.layers.BatchNormalization()(h)
    h = activations.leaky_relu(h, negative_slope=0.2)

    h = keras.layers.Dense(1024)(h)
    h = keras.layers.BatchNormalization()(h)
    h = activations.leaky_relu(h, negative_slope=0.2)

    h = keras.layers.Dense(output_size)(h)
    y = activations.tanh(h)
    return keras.Model(inputs=inputs, outputs=y, name="generator")

def create_cond_generator(latent_size=100, condition_size=10, output_size=784):
    # latent (100) -> linear (128)
    latent_inputs = keras.Input(shape=(latent_size, ))
    condition_inputs = keras.Input(shape=(condition_size, ))

    h = keras.layers.concatenate([latent_inputs, condition_inputs])
    h = keras.layers.Dense(128)(h)
    h = activations.leaky_relu(h, negative_slope=0.2)

    h = keras.layers.Dense(256)(h)
    h = keras.layers.BatchNormalization()(h)
    h = activations.leaky_relu(h, negative_slope=0.2)

    h = keras.layers.Dense(512)(h)
    h = keras.layers.BatchNormalization()(h)
    h = activations.leaky_relu(h, negative_slope=0.2)

    h = keras.layers.Dense(1024)(h)
    h = keras.layers.BatchNormalization()(h)
    h = activations.leaky_relu(h, negative_slope=0.2)

    h = keras.layers.Dense(output_size)(h)
    y = activations.tanh(h)
    return keras.Model(inputs=[latent_inputs, condition_inputs], outputs=y, name="cond-generator")

def create_cond_discriminator(input_shape, condition_size=10, num_classses=1):
    inputs = keras.Input(shape=input_shape)
    condition_inputs = keras.Input(shape=(condition_size, ))
    h = keras.layers.concatenate([inputs, condition_inputs])

    h = keras.layers.Dense(512)(h)
    h = activations.leaky_relu(h, negative_slope=0.2)
    h = keras.layers.Dense(256)(h)
    h = activations.leaky_relu(h, negative_slope=0.2)
    y = keras.layers.Dense(num_classses)(h)
    return keras.Model(inputs=[inputs, condition_inputs], outputs=y, name="cond-discriminator")

def create_dcdiscriminator(input_shape, ndf):
    """Create a discriminator model for DCGAN.
    Args:
        ndf: int, number of discriminator filters
        nc: int, number of channels
    
    Returns:
        keras.Model
    """

    inputs = keras.Input(shape=input_shape)

    h = keras.layers.Conv2D(ndf, kernel_size=(4, 4), strides=(2, 2), padding="same")(inputs)
    h = keras.layers.LeakyReLU(0.2)(h)

    h = keras.layers.Conv2D(ndf * 2, kernel_size=(4, 4), strides=(2, 2), padding="same")(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.LeakyReLU(0.2)(h)

    h = keras.layers.Conv2D(ndf * 4, kernel_size=(4, 4), strides=(2, 2), padding="same")(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.LeakyReLU(0.2)(h)

    h = keras.layers.GlobalMaxPool2D()(h)
    y = keras.layers.Dense(1)(h)
    return keras.Model(inputs=inputs, outputs=y, name="dcdiscriminator")

def create_dcgenerator(nz, ngf, nc):
    """Create a generator model for DCGAN.
    Args:
        nz: int, latent dimension
        ngf: int, number of generator filters
        nc: int, number of channels
    
    Returns:
        keras.Model
    """
    # inputs = keras.Input(shape=(1, 1, nz))
    inputs = keras.Input(shape=(nz, ))

    # reshape to (1, 1, nz)
    r_inputs = keras.layers.Reshape((1, 1, nz))(inputs)

    # state size: 4 x 4 x (ngf*8)
    h = keras.layers.Conv2DTranspose(
        filters=ngf*8, kernel_size=(4, 4), strides=(1, 1), padding="valid", use_bias=False
    )(r_inputs)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    # state size: 8 x 8 x (ngf*4)
    h = keras.layers.Conv2DTranspose(
        filters=ngf*4, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False
    )(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    # state size: 16 x 16 x (ngf*2)
    h = keras.layers.Conv2DTranspose(ngf * 2, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False)(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    # # state size: (ngf*2) x 16 x 16
    # h = keras.layers.Conv2DTranspose(ngf, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False)(h)
    # h = keras.layers.BatchNormalization()(h)
    # h = keras.layers.ReLU()(h)

    # state size: 32 x 32 x nc
    h = keras.layers.Conv2DTranspose(nc, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False)(h)
    y = keras.layers.Activation("tanh")(h)
    return keras.Model(inputs=inputs, outputs=y, name="dcgenerator")

def create_cond_dcdiscriminator(input_shape, ndf, ncond):
    """Create a discriminator model for DCGAN.
    Args:
        ndf: int, number of discriminator filters
        nc: int, number of channels
    
    Returns:
        keras.Model
    """

    nc = input_shape[-1]
    in_shape = (input_shape[0], input_shape[1], nc+ncond)

    inputs = keras.Input(shape=in_shape)

    h = keras.layers.Conv2D(ndf, kernel_size=(4, 4), strides=(2, 2), padding="same")(inputs)
    h = keras.layers.LeakyReLU(0.2)(h)

    h = keras.layers.Conv2D(ndf * 2, kernel_size=(4, 4), strides=(2, 2), padding="same")(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.LeakyReLU(0.2)(h)

    h = keras.layers.Conv2D(ndf * 4, kernel_size=(4, 4), strides=(2, 2), padding="same")(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.LeakyReLU(0.2)(h)

    h = keras.layers.GlobalMaxPool2D()(h)
    y = keras.layers.Dense(1)(h)
    return keras.Model(inputs=inputs, outputs=y, name="dcdiscriminator")

def create_cond_dcgenerator(nz, ngf, nc, ncond):
    """Create a generator model for DCGAN.
    Args:
        nz: int, latent dimension
        ngf: int, number of generator filters
        nc: int, number of channels
        ncond: int, number of condition
    
    Returns:
        keras.Model
    """
    # inputs = keras.Input(shape=(1, 1, nz))
    latent_inputs = keras.Input(shape=(nz, ))
    condition_inputs = keras.Input(shape=(ncond, ))

    all_inputs = keras.layers.concatenate([latent_inputs, condition_inputs])

    # reshape to (1, 1, nz)
    r_inputs = keras.layers.Reshape((1, 1, nz+ncond))(all_inputs)

    # state size: 4 x 4 x (ngf*8)   
    h = keras.layers.Conv2DTranspose(
        filters=ngf*8, kernel_size=(4, 4), strides=(1, 1), padding="valid", use_bias=False
    )(r_inputs)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    # state size: 8 x 8 x (ngf*4)
    h = keras.layers.Conv2DTranspose(
        filters=ngf*4, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False
    )(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    # state size: 16 x 16 x (ngf*2)
    h = keras.layers.Conv2DTranspose(ngf * 2, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False)(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    # # state size: (ngf*2) x 16 x 16
    # h = keras.layers.Conv2DTranspose(ngf, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False)(h)
    # h = keras.layers.BatchNormalization()(h)
    # h = keras.layers.ReLU()(h)

    # state size: 32 x 32 x nc
    h = keras.layers.Conv2DTranspose(nc, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False)(h)
    y = keras.layers.Activation("tanh")(h)
    return keras.Model(inputs=[latent_inputs, condition_inputs], outputs=y, name="dcgenerator")






