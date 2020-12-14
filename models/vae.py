# Importing all the libraries and functions required for defining the 
# Encoder and Decoder
from keras.callbacks.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Dense, Activation, Reshape, Lambda, LeakyReLU, Dropout
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import plot_model
import numpy as np
import os
import pickle


from utils.callbacks import Customcallback, step_decay_schedule

class VariationalAutoEncoder():

    def __init__(self, input_dim,
    encoder_filters,
    encoder_kernel_size,
    encoder_strides,
    decoder_filters,
    decoder_kernel_size,
    decoder_strides,
    z_dim,
    use_batchnorm = False,
    use_dropout = False):

        self.name = 'variational_autoencoder'

        self.input_dim = input_dim
        self.encoder_filters = encoder_filters
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_strides = encoder_strides
        self.decoder_filters = decoder_filters
        self.decoder_kernel_size = decoder_kernel_size
        self.decoder_strides = decoder_strides
        self.z_dim = z_dim
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.encoder_layers = len(encoder_filters)
        self.decoder_layers = len(decoder_filters)


        self.build()

    def build(self):
        
        # The ENCODER
        encoder_input = Input(shape=self.input_dim, name = 'encoder_input')

        x = encoder_input

        for i in range(self.encoder_layers):
            conv_layer = Conv2D(filters = self.encoder_filters[i],
                                kernel_size = self.encoder_kernel_size[i],
                                strides = self.encoder_strides[i],
                                padding = 'same',
                                name = 'encoder_conv_' + str(i))

            x = conv_layer(x)

            if self.use_batchnorm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if self.use_dropout:
                x = Dropout(0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)

        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)

        self.mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):

            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var/2) * epsilon

        encoder_output = Lambda(sampling,name = 'encoder_output')([self.mu,self.log_var])

        self.encoder = Model(encoder_input,encoder_output)

        # The DECODER

        decoder_input = Input(shape = (self.z_dim,),name = 'decoder_input')
            
        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.decoder_layers):
            transpose_conv = Conv2DTranspose(filters = self.decoder_filters[i],
                                            kernel_size = self.decoder_kernel_size[i],
                                            strides = self.decoder_strides[i],
                                            padding = 'same',
                                            name = 'transpose_conv_' + str(i))

            x = transpose_conv(x)

            if i <self.decoder_layers - 1:
                if self.use_batchnorm:
                    x = BatchNormalization()(x)
                    
                x = LeakyReLU()(x)

                if self.use_dropout:
                    x = Dropout(0.25)(x)
            else:
                x = Activation('sigmoid')(x)

        decoder_output = x

        self.decoder = Model(decoder_input,decoder_output)

        # THE FULL MODEL

        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)

    def compile(self, learning_rate, r_loss_factor):

        self.learning_rate = learning_rate

        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
            return r_loss * r_loss_factor

        def vae_kl_loss(y_true, y_pred):
            kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis=1)
            return kl_loss

        def vae_loss(y_true,y_pred):
            r_loss = vae_r_loss(y_true,y_pred)
            kl_loss = vae_kl_loss(y_true,y_pred)

            return r_loss + kl_loss

        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer,loss = vae_loss, metrics = [vae_r_loss,vae_kl_loss])

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder,'weights'))
            os.makedirs(os.path.join(folder,'images'))
        
        with open(os.path.join(folder, 'params.pkl'),'wb') as f:
            pickle.dump([self.input_dim,
                        self.encoder_filters,
                        self.encoder_kernel_size,
                        self.encoder_strides,
                        self.decoder_filters,
                        self.decoder_kernel_size,
                        self.decoder_strides,
                        self.z_dim,
                        self.use_batchnorm,
                        self.use_dropout],f)

        self.plot_model(folder)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=100, initial_epoch=0, lr_decay=1):
        
        custom_callback = Customcallback(run_folder, print_every_n_batches,initial_epoch,self)
        lr_schedule = step_decay_schedule(initial_lr = self.learning_rate, decay_factor=lr_decay, step_size = 1)
        checkpoint_filepath=os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only= True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'),save_weights_only=True, verbose=1)

        callbacks_list = [checkpoint1,checkpoint2,custom_callback,lr_schedule]

        self.model.fit(x_train,x_train,
                        batch_size=batch_size,
                        shuffle=True, 
                        epochs= epochs,
                        initial_epoch=initial_epoch,
                        callbacks=callbacks_list)

    def train_with_generator(self, data_flow, epochs, steps_per_epoch, run_folder, print_every_n_batches = 100, initial_epoch = 0, lr_decay = 1, ):

        custom_callback = Customcallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath=os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only = True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only = True, verbose=1)

        callbacks_list = [checkpoint1, checkpoint2, custom_callback, lr_sched]

        self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                
        self.model.fit_generator(
            data_flow
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
            , callbacks = callbacks_list
            , steps_per_epoch=steps_per_epoch 
            )

    
    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.encoder, to_file=os.path.join(run_folder ,'viz/encoder.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.decoder, to_file=os.path.join(run_folder ,'viz/decoder.png'), show_shapes = True, show_layer_names = True)
