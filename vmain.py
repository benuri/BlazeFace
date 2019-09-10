import tensorflow as tf
import numpy as np 
import cv2
import pickle
import glob 
import os 
import time 
import argparse
from tqdm import tqdm
import logging, coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)
coloredlogs.install()

from network import network 
from loss import smooth_l1_loss
from utils import get_iou
from dataloader import dataloader
class BlazeFace():
    
    def __init__(self, config):
        self.channels = 3
        self.input_shape = (config.input_shape, config.input_shape, self.channels)
        self.feature_extractor = network(self.input_shape)
        
        self.n_boxes = [2, 6] # 2 for 16x16, 6 for 8x8
        
        self.model = self.build_model()
        
        if config.train:
            self.batch_size = config.batch_size
            self.nb_epoch = config.nb_epoch
            
        self.checkpoint_path = config.checkpoint_path
        self.numdata = config.numdata
        
    def build_model(self):
        
        model = self.feature_extractor
        
        # Since we are only interested in face or not, the output confidence is a vector with one element. (Sigmoid function is taken because it is about one.)
        # 16x16 bounding box - Confidence, [batch_size, 16, 16, 2]
        bb_16_conf = tf.keras.layers.Conv2D(filters=self.n_boxes[0] * 1, 
                                            kernel_size=3, 
                                            padding='same', 
                                            activation='relu', name='bb_16_conf')(model.output[0])
        # reshape [batch_size, 16**2 * #bbox(2), 1]
        bb_16_conf_reshaped = tf.keras.layers.Reshape((16**2 * self.n_boxes[0], 1), name='bb_16_conf_reshaped')(bb_16_conf)
        
        # 8 x 8 bounding box - Confindece, [batch_size, 8, 8, 6]
        bb_8_conf = tf.keras.layers.Conv2D(filters=self.n_boxes[1] * 1, 
                                            kernel_size=3, 
                                            padding='same', 
                                            activation='relu', name='bb_8_conf')(model.output[1])
        # reshape [batch_size, 8**2 * #bbox(6), 1]
        bb_8_conf_reshaped = tf.keras.layers.Reshape((8**2 * self.n_boxes[1], 1), name='bb_8_conf_reshaped')(bb_8_conf)
        # Concatenate confidence prediction 
        
        # shape : [batch_size, 896, 1]
        conf_of_bb = tf.keras.layers.Concatenate(axis=1, name='conf_of_bb_concat')([bb_16_conf_reshaped, bb_8_conf_reshaped])
        
        conf_of_bb_flatten = tf.keras.layers.Flatten()(conf_of_bb)
        clf = tf.keras.layers.Dense(1, activation='sigmoid', name='clf_output')(conf_of_bb_flatten)
        
        
        # 16x16 bounding box - loc [x, y, w, h]
        bb_16_loc = tf.keras.layers.Conv2D(filters=self.n_boxes[0] * 4,
                                            kernel_size=3, 
                                            padding='same', name='bb_16_loc')(model.output[0])
        # [batch_size, 16**2 * #bbox(2), 4]
        bb_16_loc_reshaped = tf.keras.layers.Reshape((16**2 * self.n_boxes[0], 4), name='bb_16_loc_reshaped')(bb_16_loc)
        
        
        # 8x8 bounding box - loc [x, y, w, h]
        bb_8_loc = tf.keras.layers.Conv2D(filters=self.n_boxes[1] * 4,
                                          kernel_size=3,
                                          padding='same', name='bb_8_loc')(model.output[1])
        bb_8_loc_reshaped = tf.keras.layers.Reshape((8**2 * self.n_boxes[1], 4), name='bb_8_loc_reshaped')(bb_8_loc)
        # Concatenate  location prediction 
        
        loc_of_bb = tf.keras.layers.Concatenate(axis=1, name='bb_output1')([bb_16_loc_reshaped, bb_8_loc_reshaped])
        
        loc_of_bb_flatten = tf.keras.layers.Flatten()(loc_of_bb)
        bb_loc_output = tf.keras.layers.Dense(4, name='bb_output')(loc_of_bb_flatten)
        
        # output_combined = tf.keras.layers.Concatenate(axis=-1, name='bb_output')([clf, bb_loc_output])
        
        # Detectors model
        return tf.keras.models.Model(model.input, [clf, bb_loc_output])
    

    def train(self):
        
        opt = tf.keras.optimizers.Adam(amsgrad=True)
        model = self.model
        model.compile(loss=['binary_crossentropy', smooth_l1_loss], optimizer=opt)
        print(model.summary())
        """ Callback """
        monitor = 'loss'
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, patience=4)

        """ Callback for Tensorboard """
        tb = tf.keras.callbacks.TensorBoard(log_dir="./logs/", update_freq='batch')

        """ Training loop """

        ## Full dataset = 2625
        STEP_SIZE_TRAIN = self.numdata // self.batch_size

        
        data_gen = dataloader(config.dataset_dir, config.label_path, self.batch_size)

        t0 = time.time()
        
        # logging.warning('data gen')
        # for d in data_gen:
        #     print(d)


        #     break

        # return None
        for epoch in range(self.nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=data_gen,
                                      steps_per_epoch= STEP_SIZE_TRAIN,
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr, tb],
                                      verbose=1,
                                      shuffle=True,
                                      use_multiprocessing=True)
            t2 = time.time()
            
            print(res.history)
            
            print('Training time for one epoch : %.1f' % ((t2 - t1)))

            if epoch % 100 == 0:
                model.save_weights(os.path.join(config.checkpoint_path, str(epoch)))

        print('Total training time : %.1f' % (time.time() - t0))



if __name__ == "__main__":

    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--input_shape', type=int, default=128)
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--nb_epoch', type=int, default=1000)
    args.add_argument('--numdata', type=int, default=3226)
    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--checkpoint_path', type=str, default="./checkpoint/")
    args.add_argument('--dataset_dir', type=str, default="./data/images/")
    args.add_argument('--label_path', type=str, default="./data/label.csv")

    config = args.parse_args()

    blazeface = BlazeFace(config)
    logging.info('Blazeface model loaded')

    if config.train:
        blazeface.train()