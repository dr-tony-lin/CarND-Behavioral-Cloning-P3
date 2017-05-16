'''
Train behavior cloning
'''
import glob
from threading import Thread
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from keras.optimizers import Adam
from matplotlib import pyplot as plt

import utils
import model

class Trainer(Thread, Callback):
    def __init__(self, config, test_split=0.1):
        '''
        Initialize the trainer
        config - the configuration
        '''
        super().__init__()
        self.config = config
        self.model = None
        self.test_model = None
        self.train_requested = False
        self.test_requested = False
        self.save_requested = False
        self.training = False
        self.testing = False
        self.running = False
        self.history = None
        self.epoch = 0

        # Get training samples
        self.images, self.data = utils.get_samples(dirs=config.dirs, flip=config.flip, all_cameras=config.all_cameras)
        # Randomly shuffle images
        self.images, self.data = shuffle(self.images, self.data)
        # Split the training samples for training and validation
        self.images, self.validate_images, self.data, self.validate_data = train_test_split(self.images,
                                                                                            self.data,
                                                                                            test_size=test_split)

        if config.test is not None: # Test is configured, get test samples
            self.test_images, self.test_data = utils.get_samples(dirs=[config.test], flip=False,
                                                                 all_cameras=config.all_cameras)
        print("Training samples: ", len(self.images))
        print("Validation samples: ", len(self.validate_images))
        if config.test:
            print("Test samples: ", len(self.test_images))

    def save_history(self, file='history.png'):
        '''
        Save history
        '''
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.savefig(file)

    def _train(self):
        '''
        Train the model
        '''
        # Create generators for training and tests
        training_image_generator = utils.DrivingDataGenerator(self.images, self.data, batch_size=self.config.batch)
        validation_image_generator = utils.DrivingDataGenerator(self.validate_images, self.validate_data,
                                                                batch_size=self.config.batch)

        if self.config.model is None: # Train new model
            optimizer = Adam(lr=self.config.lr)
            self.model = model.create_nvidia_model(input_shape=(3, 160, 320), cropping=((50, 20), (0, 0)),
                                                   dropout=self.config.drr)
            self.model = model.create_training_model(self.model, optimizer=optimizer)
        else: # Continue from the given training model
            if self.config.cont:
                print("Continue training from {}".format(self.config.model))
                self.model = model.load_checkpoint(self.config.model)
            else:
                print("Train from {}".format(self.config.model))
                optimizer = Adam(lr=self.config.lr)
                self.model = model.create_nvidia_model(input_shape=(3, 160, 320), cropping=((50, 20), (0, 0)),
                                                       dropout=self.config.drr)
                self.model = model.create_training_model(self.model, optimizer=optimizer)
                model.load_checkpoint(self.config.model, self.model)

        # Train the model
        start_time = time.time()
        self.history = model.train_model(self.model, training_image_generator,
                                         validation_image_generator, len(self.images)/self.config.batch,
                                         len(self.validate_images)/self.config.batch, self.config, self)
        print("Training time: {0:6} seconds".format(int(time.time()-start_time)))
        if self.config.analytics is not None:
            self.save_history(self.config.analytics)
            print("Training history saved in {}".format(self.config.analytics))

    def _test(self):
        '''
        Test trained models
        '''
        if self.test_images: # Test training results
            for checkpoint in glob.glob(self.config.checkpoint + "*.h5"):
                test_image_generator = utils.DrivingDataGenerator(self.test_images, self.test_data,
                                                                  batch_size=self.config.batch)
                start_time = time.time()
                print("Testing: ", checkpoint)
                self.test_model = model.load_checkpoint(checkpoint)
                results = model.test_model(self.test_model, test_image_generator,
                                           len(self.test_images)/self.config.batch)
                print("Testing time: {0:6}, result: {1}".format(int(time.time()-start_time), results))

    def train(self):
        '''
        Schedule training
        '''
        if self.training:
            return False
        self.train_requested = True
        return True

    def test(self):
        '''
        Schedule test
        '''
        if self.testing:
            return False
        self.test_requested = True
        return True

    def stop(self):
        '''
        Stop current operation
        '''
        self.model.stop_training = True

    def end(self):
        '''
        End the thread
        '''
        self.stop()
        self.running = False

    def save(self):
        '''
        Save current model
        '''
        if self.training:
            self.save_requested = True
        elif self.epoch == 0:
            name = "{0}.json".format(self.config.checkpoint)
            json = self.model.to_json()
            with open(name, 'w') as file:
                file.write(json)
        else:
            name = "{0}-save-{1}".format(self.config.checkpoint, self.epoch)
            model.save_checkpoint(self.model, name, False)

    def on_epoch_end(self, epoch, logs=None):
        '''
        Callback at each epoch end. Save the checkpoint if the accuracy is above the threshold
        '''
        self.epoch = epoch + 1
        if self.save_requested or (logs is not None and logs['loss'] <= (1. - self.config.accept)):
            self.save_requested = False
            name = "{0}-{1}".format(self.config.checkpoint, self.epoch)
            print("Saving: {0}, accuracy: {1}".format(name, 1-logs['loss']), logs)
            # Saving Lambda layer on Windows 10 annuiversary edition will fail, so only save weights
            #if 'Windows' in platform.platform() and platform.release() == '10':
            #    save_checkpoint(self.model, name, True)
            #else:
            model.save_checkpoint(self.model, name, False)
        else:
            print("Accuracy: {0}".format(1 - logs['loss']), logs)

    def run(self):
        '''
        The thread loop
        '''
        self.running = True
        while self.running:
            if self.train_requested:
                self.train_requested = False
                self.training = True
                self._train()
                self.training = False
            elif self.test_requested:
                self.train_requested = False
                self.testing = True
                self._test()
                self.testing = False
            time.sleep(1)
