'''
Model helper for behavior cloning using a CNN based on Nividia's sel-driviong car CNN model.
It also use Leaky Relu and ELU for activation. The loss function for training is mean sequre error.
'''
from keras.callbacks import Callback, ReduceLROnPlateau, TensorBoard
from keras.engine.topology import Layer
from keras.layers import Dense, Conv2D, Flatten, Input, Dropout
from keras.layers.convolutional import Cropping2D
from keras.layers.advanced_activations import LeakyReLU
from keras import metrics
from keras.models import Model, load_model

class Normalization2D(Layer):
    '''
    Custom layer for image normalization. This is mainly for Windows 10 annuiversary edition where Lambda
    layer will cause save to fail
    '''
    def call(self, inputs, **kwargs):
        return (inputs / 255.0) - 0.5

    def compute_output_shape(self, input_shape):
        return input_shape

def create_nvidia_model(input_shape, cropping=None, dropout=0.5, leakyRelu=0.02):
    '''
    Create Nvidia self-driving car CNN model consists of 5 convolutional layers and 4 network layers
    '''
    # Preprocessing and normalization
    inputs = Input(shape=input_shape)
    cropping = Cropping2D(cropping=cropping)(inputs)
    # On Windows 10 anniversary Lambda will fail to serialize, use custom layer to avoid this issue
    normalization = Normalization2D()(cropping) #Lambda(lambda x: (x / 255.0) - 0.5)(cropping)

    # Layer 1
    conv1 = Conv2D(filters=24, kernel_size=5, strides=(2, 2), padding='valid',
                   use_bias=True, kernel_initializer='glorot_normal')(normalization)
    activation1 = LeakyReLU(alpha=leakyRelu)(conv1)

    # Layer 2
    conv2 = Conv2D(filters=36, kernel_size=5, strides=(2, 2), padding='valid',
                   use_bias=True, kernel_initializer='glorot_normal')(activation1)
    activation2 = LeakyReLU(alpha=leakyRelu)(conv2)

    # Layer 3
    conv3 = Conv2D(filters=48, kernel_size=5, strides=(2, 2), padding='valid',
                   use_bias=True, kernel_initializer='glorot_normal')(activation2)
    activation3 = LeakyReLU(alpha=leakyRelu)(conv3)

    # Layer 4
    conv4 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid',
                   use_bias=True, kernel_initializer='glorot_normal')(activation3)
    activation4 = LeakyReLU(alpha=leakyRelu)(conv4)

    # Layer 5
    conv5 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid',
                   use_bias=True, kernel_initializer='glorot_normal')(activation4)
    activation5 = LeakyReLU(alpha=leakyRelu)(conv5)

    flat = Flatten()(activation5)

    # Layer 6
    dense6 = Dense(units=100, use_bias=True, kernel_initializer='glorot_normal',
                   activation='elu')(flat)
    if dropout is not None:
        dense6 = Dropout(dropout)(dense6)

    # Layer 7
    dense7 = Dense(units=50, use_bias=True, kernel_initializer='glorot_normal',
                   activation='elu')(dense6)
    if dropout is not None:
        dense7 = Dropout(dropout)(dense7)

    # Layer 8
    dense8 = Dense(units=10, use_bias=True, kernel_initializer='glorot_normal',
                   activation='elu')(dense7)
    if dropout is not None:
        dense8 = Dropout(dropout)(dense8)

    # Layer 9
    dense9 = Dense(units=1, use_bias=True, kernel_initializer='glorot_normal',
                   activation='elu')(dense8)

    return Model(inputs=inputs, outputs=dense9)

def save_checkpoint(model, checkpoint, weight_only=True):
    '''
    Save the checkpoint
    model: the model
    checkpoint: the checkpoint name.
    weight_only: True to save only weight, the saved weight will be in checkpoint_weights.h5,
    otherwise, the model and weight will be save to checkpoint.h5
    '''
    if weight_only:
        model.save_weights(checkpoint + "_weights.h5")
    else:
        model.save(checkpoint + ".h5")

def load_checkpoint(checkpoint, model=None):
    '''
    Load checkpoint
    checkpoint: the checkpoint name
    model: the model, if None, model will be loaded from checkpoint.h5, otherwise, model's weights
    will be loaded from checkpoint_weights.h5
    '''
    if model is None:
        return load_model(checkpoint, custom_objects={'Normalization2D': Normalization2D})
    else:
        return model.load_weights(checkpoint)

def create_training_model(model, optimizer='ADAM', loss='mean_squared_error'):
    '''
    Create a training model
    model: the bare model
    optimizer: the optimizer
    loss: the loss function
    '''
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics.mae, metrics.mean_squared_error])
    return model

def create_test_model(model, loss='mean_squared_error'):
    '''
    Create a training model
    model: the bare model
    optimizer: the optimizer
    loss: the loss function
    '''
    model.compile(optimizer='ADAM', loss=loss, metrics=[metrics.mean_squared_error])
    return model

def train_model(model, train_generator, validation_generator, train_steps, validation_steps, config, callbacks=None):
    '''
    Train the model

    Arguments:
    model - the model
    train_generator - training sample generator
    validation_generator - validation sample generator
    train_steps - the number of batch steps in each epoch
    validation_steps - the number of validation batch steps
    config - the configuration
    '''
    default_callbcks = [ # Extra training callbacks
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=0, mode='auto', epsilon=0.001,
                          cooldown=0, min_lr=1e-6),
        TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0,
                    embeddings_layer_names=None, embeddings_metadata=None)
    ]

    if callbacks is None:
        callbacks = default_callbcks
    elif isinstance(callbacks, Callback):
        callbacks = default_callbcks + [callbacks]
    else:
        assert isinstance(callbacks, list), "Invalid callbacks, must be a list of Callback, or a Callback"

    # Training
    history = model.fit_generator(train_generator, validation_data=validation_generator, epochs=config.epochs,
                                  steps_per_epoch=train_steps,
                                  validation_steps=validation_steps,
                                  verbose=2, callbacks=callbacks)
    return history

def test_model(model, test_generator, test_steps):
    '''
    Test the model

    Arguments:
    model - the model
    test_generator - the test sample generator
    test_steps - the number of test batch steps
    '''
    model = create_test_model(model)
    return model.evaluate_generator(generator=test_generator, steps=test_steps)
