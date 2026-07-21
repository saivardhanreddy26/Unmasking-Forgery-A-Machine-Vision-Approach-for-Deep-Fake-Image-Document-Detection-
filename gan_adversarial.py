import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, Conv2DTranspose, LeakyReLU, 
    BatchNormalization, Reshape, Flatten, Input, Dropout
)
from tensorflow.keras.optimizers import Adam
import numpy as np

class AdversarialGAN:
    """
    Generative Adversarial Network for adversarial training
    Generates challenging forgery examples to improve detector robustness
    """
    
    def __init__(self, img_shape=(256, 256, 3), latent_dim=100):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        
        # Build and compile the discriminator (uses main detector)
        self.discriminator = None
        
        # Build the generator
        self.generator = self.build_generator()
        
        # Build the combined GAN model
        self.gan = None
    
    def build_generator(self):
        """
        Build the generator network that creates synthetic forgeries
        """
        model = Sequential([
            # Input: noise vector
            Dense(16 * 16 * 512, input_dim=self.latent_dim),
            LeakyReLU(alpha=0.2),
            Reshape((16, 16, 512)),
            
            # Upsample to 32x32
            Conv2DTranspose(256, (5, 5), strides=(2, 2), 
                          padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            
            # Upsample to 64x64
            Conv2DTranspose(128, (5, 5), strides=(2, 2), 
                          padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            
            # Upsample to 128x128
            Conv2DTranspose(64, (5, 5), strides=(2, 2), 
                          padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            
            # Upsample to 256x256
            Conv2DTranspose(32, (5, 5), strides=(2, 2), 
                          padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            
            # Output layer
            Conv2D(3, (5, 5), padding='same', activation='tanh')
        ], name='Generator')
        
        return model
    
    def build_discriminator(self, base_detector):
        """
        Use the main forgery detector as discriminator
        
        Args:
            base_detector: The CNN-ANN pipeline model
        """
        self.discriminator = base_detector
        self.discriminator.trainable = True
    
    def build_gan(self):
        """
        Build the combined GAN model
        Generator -> Discriminator (frozen during generator training)
        """
        # Freeze discriminator weights during generator training
        self.discriminator.trainable = False
        
        # GAN input (noise)
        gan_input = Input(shape=(self.latent_dim,))
        
        # Generate fake image
        generated_image = self.generator(gan_input)
        
        # Discriminator determines validity
        validity = self.discriminator(generated_image)
        
        # Combined model
        self.gan = Model(gan_input, validity, name='GAN')
        
        # Compile GAN
        self.gan.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )
    
    def train_adversarial(self, real_images, epochs=100, 
                         batch_size=32, sample_interval=10):
        """
        Adversarial training loop
        
        Args:
            real_images: Real training images
            epochs: Number of training epochs
            batch_size: Batch size
            sample_interval: Interval for sampling generated images
        """
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select random batch of real images
            idx = np.random.randint(0, real_images.shape[0], batch_size)
            real_imgs = real_images[idx]
            
            # Generate fake images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise, verbose=0)
            
            # Train discriminator
            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch(real_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Train generator (wants discriminator to mistake images as real)
            self.discriminator.trainable = False
            g_loss = self.gan.train_on_batch(noise, valid)
            
            # Print progress
            if epoch % sample_interval == 0:
                print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss[0]:.4f} | "
                     f"G Loss: {g_loss:.4f} | D Acc: {100*d_loss[1]:.2f}%")
    
    def generate_adversarial_examples(self, num_samples=100):
        """
        Generate adversarial forgery examples
        
        Args:
            num_samples: Number of examples to generate
        
        Returns:
            Generated adversarial images
        """
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        generated_images = self.generator.predict(noise, verbose=0)
        
        # Scale from [-1, 1] to [0, 255]
        generated_images = (generated_images + 1) * 127.5
        generated_images = generated_images.astype(np.uint8)
        
        return generated_images
    
    def save_generator(self, path):
        """Save generator model"""
        self.generator.save(path)
    
    def load_generator(self, path):
        """Load generator model"""
        self.generator = tf.keras.models.load_model(path)
