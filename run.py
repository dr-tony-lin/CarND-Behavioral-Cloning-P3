'''
This kis the main entry point of training
'''
import os
from config import config
import utils
from train import Trainer

# Create trainer thread
trainer_thread = Trainer(config)

def command_handler(command):
    '''
    Handler user command
    '''
    if command == 'stop':
        if trainer_thread.training:
            print("Stop training when the current epoch is completed ...")
            trainer_thread.stop()
    elif command == 'train':
        if trainer_thread.training:
            print("Training is already in progress!")
        else:
            if trainer_thread.testing:
                print("Training will be started after test is completed.")
            else:
                print("Start training ...")
            trainer_thread.train()
    elif command == 'test':
        if trainer_thread.testing:
            print("Test is already in progress!")
        else:
            if trainer_thread.training:
                print("Test will be started after training is completed.")
            else:
                print("Start testing ...")
            trainer_thread.test()
    elif command.find('accept') == 0:
        config.accept = float(command[len('accept'):].strip())
        print("Accept accuracy over {0}".format(config.accept))
    elif command == 'save':
        if trainer_thread.training:
            print("The current epoch will be saved upon completion.")
        elif trainer_thread.epoch == 0:
            print("Training has not been started, save the model only ...")
        trainer_thread.save()
    elif command == 'exit':
        print("Exiting ...")
        trainer_thread.end()
        os._exit(0)
    else:
        print("Unknown command: {}!".format(command))

input_thread = utils.accept_inputs(command_handler)
trainer_thread.start()
trainer_thread.train()
trainer_thread.join()