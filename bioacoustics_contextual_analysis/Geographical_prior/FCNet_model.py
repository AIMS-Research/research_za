# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:06:45 2022

@author: ljeantet

'''
Script related to the article "Empowering Deep Learning Acoustic Classifiers with Human-like Ability 
to Utilize Contextual Information for Wildlife Monitoring" by Jeantet and Dufourq.

code adapted from @author:  Copyright 2021 Fagner Cunha

from github https://github.com/alcunha/geo_prior_tf/blob/master/geo_prior/dataloader.py

"""

import tensorflow as tf


import time
import datetime
import os
import shutil


def create_new_folder(model_out, model_name):
    
    now = datetime.datetime.now()
    today=now.strftime("%Y_%m_%d")
    dir_out=model_out+"model="+today+"_"+ model_name
    
   
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
        print("we clear the directory:",dir_out)
    else:
        print("we create the directory:",dir_out)

    
    os.makedirs(dir_out)
    return dir_out


def _create_res_layer(inputs, embed_dim, use_bn=False):
    x = tf.keras.layers.Dense(embed_dim)(inputs)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(embed_dim)(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    outputs = tf.keras.layers.add([inputs, x])

    return outputs

def _create_loc_encoder(inputs, embed_dim, num_res_blocks, use_bn=False):
    x = tf.keras.layers.Dense(embed_dim)(inputs)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    for _ in range(num_res_blocks):
        x = _create_res_layer(x, embed_dim)

    return x

def _create_FCNet(num_inputs,
                  num_classes,
                  embed_dim,
                  num_res_blocks=4,
                  use_bn=False):
    inputs = tf.keras.Input(shape=(num_inputs,))
    loc_embed = _create_loc_encoder(inputs, embed_dim, num_res_blocks, use_bn)
    class_embed_layer = tf.keras.layers.Dense(num_classes,
                                            activation='sigmoid',
                                            use_bias=False)
    class_embed = class_embed_layer(loc_embed)


    outputs = [class_embed]


    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model, class_embed_layer


def lr_scheduler(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * lr_decay
  
    
class FCNet(tf.keras.Model):
    def __init__(self, num_inputs, embed_dim, num_classes, rand_sample_generator,
                num_res_blocks=4, use_bn=False):
        super(FCNet, self).__init__()
    
        model, class_embed = _create_FCNet(
                               num_inputs, num_classes, embed_dim, 
                               num_res_blocks=num_res_blocks, use_bn=use_bn)
        self.model = model
        self.class_embed = class_embed
        self.rand_sample_generator = rand_sample_generator
    

    def call(self, inputs):
        return self.model(inputs)

  
    def train_step(self, data):
        
        x, y = data
        batch_size = tf.shape(x)[0]
        
        y_class_true = y
    
        rand_samples = self.rand_sample_generator.get_rand_samples(batch_size)
        combined_inputs = tf.concat([x, rand_samples], axis=0)
        
        rand_labels = tf.zeros(shape=y_class_true.shape)
        
        with tf.GradientTape() as tape:
            
            y_pred = self(combined_inputs, training=True)
            total_loss = 0
            
            obj_loss = self.compiled_loss(y_class_true, y_pred[:batch_size], regularization_losses=self.losses)
            obj_loss_rand = self.compiled_loss(rand_labels, y_pred[batch_size:],regularization_losses=self.losses)
            total_loss = total_loss + obj_loss + obj_loss_rand 



        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
     
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = total_loss
        metrics['obj_loss'] = obj_loss
        metrics['obj_loss_rand'] = obj_loss_rand

        return metrics

    def test_step(self, data):
        x , y = data
        y_pred_class = self(x, training=False)
        y_true = y

        loss = self.compiled_loss(y_true, y_pred_class, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y_true, y_pred_class)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss
        metrics['obj_loss'] = loss

        return metrics

