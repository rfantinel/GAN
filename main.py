from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Convolution2D, MaxPooling2D, Deconvolution2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np



class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)

        input_layer = Input(shape=noise_shape)

        M = Reshape((10, 10, 1))(input_layer)
        M = Deconvolution2D(filters=64, kernel_size=(3, 3))(M) #64@12x12
        M = UpSampling2D(size=(2,2))(M) #64@24x24
        M = LeakyReLU(alpha=0.2)(M)
        M = BatchNormalization(momentum=0.8)(M)
        M = Convolution2D(filters=32, kernel_size=(3, 3))(M) #32@22x22
        M = Deconvolution2D(filters=16, kernel_size=(3, 3))(M) #16@24x24
        M = Deconvolution2D(filters=8, kernel_size=(3, 3))(M) #8@26x26
        M = LeakyReLU(alpha=0.2)(M)
        M = BatchNormalization(momentum=0.8)(M)
        M = Convolution2D(filters=4, kernel_size=(3, 3))(M) #4@24x24
        M = Deconvolution2D(filters=2, kernel_size=(3, 3))(M) #2@26x26
        
        output_layer = Deconvolution2D(filters=1, kernel_size=(3, 3), activation='tanh')(M) #1@28x28
        
        model = Model(input_layer, output_layer)

        print(model.summary())

        return model


    def build_discriminator(self):

        # img = 28x28x1
        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        input_layer = Input(shape=img_shape)

        M = Convolution2D(filters=32, kernel_size=(3,3), strides=(1,1))(input_layer) #32@26x26
        M = MaxPooling2D(pool_size=(2,2))(M) #32@13x13
        
        M = Convolution2D(filters=64, kernel_size=(3,3), strides=(1,1))(M) #64@11x11
        M = MaxPooling2D(pool_size=(2,2))(M) #64@5x5

        M = Convolution2D(filters=128, kernel_size=(3,3), strides=(1,1))(M) #128@3x3

        M = Flatten() (M)
        M = Dense(512, init='normal')(M)
        M = LeakyReLU(alpha=0.2) (M)
        M = Dense(256, init='normal')(M)
        M = LeakyReLU(alpha=0.2) (M)

        output_layer = Dense(1, init='normal', activation='sigmoid')(M)

        model = Model(input_layer, output_layer)

        print(model.summary())

        return model

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=128, save_interval=200)