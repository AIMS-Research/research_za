# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 08:25:52 2022

@author: ljeantet


Script related to the article "Empowering Deep Learning Acoustic Classifiers with Human-like Ability 
to Utilize Contextual Information for Wildlife Monitoring" by Jeantet and Dufourq.

code adapted from @author:  Copyright 2021 Fagner Cunha

from github https://github.com/alcunha/geo_prior_tf/blob/master/geo_prior/dataloader.py


"""

import pandas as pd
import numpy as np
import os
import json
import math
import tensorflow as tf

from keras.utils.np_utils import to_categorical



class Prior_geo_MetaData_Generator:
    
    def __init__(self, folder, database, loc_encode, dico=False, batch_size=8,          
                  max_instances_per_class=-1, is_training=False,
                  num_classes=None, batch_drop_remainder=True ):
        
        
        self.folder = folder
        self.database=database
        self.database_file=self.folder+self.database
        self.dico=dico
        
        self.is_training = is_training
        self.batch_size=batch_size
        self.batch_drop_remainder = batch_drop_remainder
        self.loc_encode=loc_encode
        self.max_instances_per_class = max_instances_per_class   
        self.num_instances = 0
        self.num_classes = num_classes
            
    def create_and_save_dictionnary(self, labels):
    
        labelName_to_labelInd={}
        labelInd_to_labelName={}
    
        for i,name in enumerate(labels):
            labelName_to_labelInd[name]=i
            labelInd_to_labelName[i]=name
    
        #save dictionnary
        with open(self.folder+"/labelName_to_labelInd_8994species.json", 'w') as f:
            json.dump(json.dumps(labelName_to_labelInd), f)
    
        with open(self.folder+"/labelInd_to_labelName_8994species.json", 'w') as f:
            json.dump(json.dumps(labelInd_to_labelName), f)
    
        return labelName_to_labelInd,labelInd_to_labelName
    
  

    def load_dico(self, path, key_int=False,print_me=False):
        with open(path) as f:
            dico_str = json.loads(json.load(f))
    
        if key_int: 
            conv_key=lambda k:int(k)
        else:
            conv_key=lambda k:k
        
        dico={conv_key(k):v for k,v in dico_str.items()}
    
        if print_me:
            print(dico)
        
        return dico

    def _get_balanced_dataset(self, metadata):
      categories_ds = []
      categories_weights = []

      for category in list(metadata.category_id.unique()):
        cat_metadata = metadata[metadata.category_id == category]
        num_instances_cat = len(cat_metadata)

        cat_ds = tf.data.Dataset.from_tensor_slices((
                        cat_metadata.id,
                        cat_metadata.lat,
                        cat_metadata.lng,
                        cat_metadata.category_id))
        
        
        cat_ds = cat_ds.shuffle(num_instances_cat).repeat()
        cat_weight = float(min(num_instances_cat, self.max_instances_per_class))
        categories_weights.append(cat_weight)      
        categories_ds.append(cat_ds)

      dataset = tf.data.experimental.sample_from_datasets(
                                                  categories_ds,
                                                  weights=categories_weights)
      self.num_instances = int(sum(categories_weights))

      return dataset

    def make_meta_dataset(self):
    
        
        #prepare database ref and load dict    
        meta=pd.read_csv(self.database_file)
        
        columns_to_keep=['id','Scientifique_name','cnt','lat','lng']
        meta=meta.loc[:,columns_to_keep]
        
        
        if self.num_classes is None:
          self.num_classes = np.unique(meta.Scientifique_name).shape[0]
        self.num_instances = meta.shape[0]
        
        
        # Create the dictionary or load it if already created
        if self.dico==True:
            labelName_to_labelInd=self.load_dico(self.folder+"/labelName_to_labelInd_8994species.json",key_int=False,print_me=False)
        else: 
            labels=np.unique(meta.Scientifique_name)
            labelName_to_labelInd, labelInd_to_labelName =self.create_and_save_dictionnary(labels)
        
        #remove lines without the lat et long information
        meta= meta.dropna(subset=['lng', 'lat'])
        
        # Associate the number id to each species 
        category_id=[labelName_to_labelInd[x] for x in meta.Scientifique_name]
        meta['category_id']=category_id
          
        # Put the data into tensor 
        if self.max_instances_per_class == -1:
          dataset = tf.data.Dataset.from_tensor_slices((
            meta.id,
            meta.lat,
            meta.lng,
            meta.category_id))

          if self.is_training:
            dataset.shuffle(self.num_instances)
        else:
          dataset = self._get_balanced_dataset(meta)

        # Functions used to process the latitude and longitudes values
        def _encode_feat(feat, encode):
            if encode == 'encode_cos_sin':
                return tf.sin(math.pi*feat), tf.cos(math.pi*feat)
            else:
                raise RuntimeError('%s not implemented' % encode)

            return feat 

        def _preprocess_data(id, lat, lng, category_id):

            lat = tf.cond(lat!=0, lambda: lat/90.0, lambda: tf.cast(0.0, tf.float64))
            lng = tf.cond(lng!=0, lambda: lng/180.0, lambda: tf.cast(0.0, tf.float64))
            lat = _encode_feat(lat, self.loc_encode)
            lng = _encode_feat(lng, self.loc_encode)

            inputs = tf.concat([lng, lat], axis=0)
            inputs = tf.cast(inputs, tf.float32)

            category_id = tf.one_hot(category_id, self.num_classes)


            outputs = category_id
            return inputs, outputs

        dataset = dataset.map(_preprocess_data)
        dataset = dataset.batch(self.batch_size,
                        drop_remainder=self.batch_drop_remainder)
        #dataset = dataset.prefetch()

        return (dataset, self.num_instances, self.num_classes)



class RandSpatioTemporalGenerator:
  def __init__(self,
               rand_type='spherical',
               loc_encode='encode_cos_sin'):
    self.rand_type = rand_type
    self.loc_encode = loc_encode


  def _encode_feat(self, feat, encode):
    if encode == 'encode_cos_sin':
      feats = tf.concat([
        tf.sin(math.pi*feat),
        tf.cos(math.pi*feat)], axis=1)
    else:
      raise RuntimeError('%s not implemented' % encode)

    return feats

  def get_rand_samples(self, batch_size):
    if self.rand_type == 'spherical':
      rand_feats = tf.random.uniform(shape=(batch_size, 3),
                                    dtype=tf.float32)
      theta1 = 2.0*math.pi*rand_feats[:,0]
      theta2 = tf.acos(2.0*rand_feats[:,1] - 1.0)
      lat = 1.0 - 2.0*theta2/math.pi
      lon = (theta1/math.pi) - 1.0
      time = rand_feats[:,2]*2.0 - 1.0

      lon = tf.expand_dims(lon, axis=-1)
      lat = tf.expand_dims(lat, axis=-1)

    else:
      raise RuntimeError('%s rand type not implemented' % self.rand_type)

    lon = self._encode_feat(lon, self.loc_encode)
    lat = self._encode_feat(lat, self.loc_encode)



    return tf.concat([lon, lat], axis=1)


       