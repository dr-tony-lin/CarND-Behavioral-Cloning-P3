'''
Initialize command argument parsing and provide config for command line options
'''
import os
import tensorflow as tf

tf.app.flags.DEFINE_string('checkpoint', os.path.expanduser('~')+'/trained/model', "Path and base name of checkpoints")
tf.app.flags.DEFINE_string('dirs', 'data', "Training sample folders, seperated by :")
tf.app.flags.DEFINE_string('test', None, "Testing sample folder")
tf.app.flags.DEFINE_string('model', None, "Continue training from the model")
tf.app.flags.DEFINE_string('analytics', "history.png", "Show training analytic")

tf.app.flags.DEFINE_integer('trainings', 1, "The number of trainings.")
tf.app.flags.DEFINE_integer('epochs', 200, "The number of epochs per train.")
tf.app.flags.DEFINE_integer('batch', 256, "The batch size.")

tf.app.flags.DEFINE_float('cr', 0.2, "The left/right cammera correction.")
tf.app.flags.DEFINE_float('lr', None, "The learning rate.")
tf.app.flags.DEFINE_float('drr', 0.5, "The dropout retention ratio.")
tf.app.flags.DEFINE_float('accept', 0.994, "The accepted training validation accuracy.")

tf.app.flags.DEFINE_bool('all_cameras', False, "True to use three cameras, False to use the center camera.")
tf.app.flags.DEFINE_bool('flip', True, "True to include flipped images")
tf.app.flags.DEFINE_bool('cont', False, "True to continue from the trained model, False to train from weights")

config = tf.app.flags.FLAGS
config.save_next = False
config.dirs = config.dirs.split(':')
dirs = []
# Expand repeated directories, e.g. a*3 will become a, a, a
for d in config.dirs:
    if '*' in d:
        d, repeats = d.split('*')
        d = d.strip()
        dirs += ([d for i in range(int(repeats.strip()))])
    else:
        dirs.append(d.strip())

config.dirs = dirs

def lrFromBatch():
    # The optimal learning rate for batch size 256 is 0.001, while the optrimal rate for batch size 16 is about 0.0001.
    # When batch size is changed the learning rate should be adjusted proportionally. The following equation uses linear
    # interpolation to decide the learning rate given batch size
    config.lr = 0.0001 + (0.001 - 0.0001) * (config.batch - 16.0)/(256.0 - 16.0)
    return config.lr

if config.lr is None:
    lrFromBatch()
