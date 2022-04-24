# -*- coding: utf-8 -*-
"""basic_Ranking_course_rec.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NjtrUW2-b3pPZVjVc8Qr7yOA891Hm2rp
"""


import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs


"""## Preparing the dataset"""

ratings_df = pd.read_csv('/content/drive/MyDrive/FYP-CRSRider/data/ramanan/filtered_review.csv')
ratings_df = ratings_df[['course_id','rating','user_displayname']]
favourable_rating_users = ratings_df['user_displayname'].value_counts().loc[lambda x: x>=3].loc[lambda x: x<100].index.values
ratings_df = ratings_df[ratings_df.user_displayname.isin(favourable_rating_users)]
ratings_df.course_id = ratings_df.course_id.astype('str')
ratings_df.rating = ratings_df.rating.astype('float32')
courses_df = pd.read_csv('/content/drive/MyDrive/FYP-CRSRider/data/ramanan/preprocessed_udemy_courses.csv')
courses_df.course_id = courses_df.course_id.astype('str')

ratings = tf.data.Dataset.from_tensor_slices(dict(ratings_df))
courses = tf.data.Dataset.from_tensor_slices(dict(courses_df))
ratings_df.loc[ratings_df.course_id.isna()]
ratings_df

for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)

for x in courses.take(1).as_numpy_iterator():
  pprint.pprint(x)

ratings = ratings.map(lambda x: {
    "course_id": x["course_id"],
    "user_id": x["user_displayname"],
    'rating': x["rating"]
})
# courses = courses.map(lambda x: x["course_id"])

"""To fit and evaluate the model, we need to split it into a training and evaluation set. In an industrial recommender system, this would most likely be done by time: the data up to time $T$ would be used to predict interactions after $T$.


In this simple example, however, let's use a random split, putting 80% of the ratings in the train set, and 20% in the test set.
"""

feature_names = ["user_id", "course_id"]

vocabularies = {}

# come up with vocab lists for string variables (doesn't really handle review text now)
# floats will automatically get converted to integers
for feature_name in feature_names:
    vocab = ratings.batch(1000000).map(lambda x: x[feature_name])
    vocabularies[feature_name] = np.unique(np.concatenate(list(vocab)))

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# course_ids = courses.batch(1_000)
# user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

# unique_courses = np.unique(np.concatenate(list(course_ids)))
# unique_users = np.unique(np.concatenate(list(user_ids)))
# # list(user_ids)[:10]

"""## Implementing a model

Choosing the architecture of our model is a key part of modelling.

Because we are building a two-tower retrieval model, we can build each tower separately and then combine them in the final model.

The task itself is a Keras layer that takes the query and candidate embeddings as arguments, and returns the computed loss: we'll use that to implement the model's training loop.

### The full model

We can now put it all together into a model. TFRS exposes a base model class (`tfrs.models.Model`) which streamlines building models: all we need to do is to set up the components in the `__init__` method, and implement the `compute_loss` method, taking in the raw features and returning a loss value.

The base model will then take care of creating the appropriate training loop to fit our model.
"""

class DCN(tfrs.Model):
    def __init__(self, use_cross_layer, deep_layer_sizes, projection_dim=None):
        super().__init__()
        
        self.embedding_dimension = 32
        
        str_features = ["user_id", "course_id"]
        # int_features = ["rating"]
        
        self._all_features = str_features
        # self._all_features = str_features + int_features
        self._embeddings = {}
        
        # compute embeddings for string features
        for feature_name in str_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [tf.keras.layers.StringLookup(
                    vocabulary=vocabulary, mask_token=None),
                 tf.keras.layers.Embedding(len(vocabulary)+1, 
                                                self.embedding_dimension)
                ])


        # vocabulary = vocabularies['rating']
        # self._embeddings['rating'] = tf.keras.Sequential([
        #     tf.keras.layers.IntegerLookup(
        #         vocabulary=vocabulary, mask_token=None),
        #     tf.keras.layers.Embedding(len(vocabulary)+1, 
        #                                  self.embedding_dimension)
        # ])
        
        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=projection_dim,
                kernel_initializer="glorot_uniform")
        else:
            self._cross_layer = None
            
        self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu")
            for layer_size in deep_layer_sizes]
        
        self._logit_layer = tf.keras.layers.Dense(1)
        
        
        self.task = tfrs.tasks.Ranking(
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")]
        )
        
    def call(self, features):
        #concat embeddings
        embeddings = []
        for feature_name in self._all_features:
            embedding_fn = self._embeddings[feature_name]
            embeddings.append(embedding_fn(features[feature_name]))
            
        x = tf.concat(embeddings, axis=1)
        
        # build cross network
        if self._cross_layer is not None:
            x = self._cross_layer(x)
            
        # build deep network
        for deep_layer in self._deep_layers:
            x = deep_layer(x)
            x = tf.keras.layers.Dropout(0.2)(x)

        return self._logit_layer(x)
    
    
    def compute_loss(self, features, training=False):
        labels = features.pop("rating")
        scores = self(features)
        return self.task(
                    labels=labels,
                    predictions=scores,
        )

"""## Fitting and evaluating

After defining the model, we can use standard Keras fitting and evaluation routines to fit and evaluate the model.

Let's first instantiate the model.
"""

learning_rate = 0.0001
lr_model = DCN(use_cross_layer=True, deep_layer_sizes=[128, 128])
lr_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

"""Then shuffle, batch, and cache the training and evaluation data."""

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

"""Then train the  model:"""

lr_model.fit(cached_train,validation_data = cached_test, epochs=3)

"""Finally, we can evaluate our model on the test set:"""

lr_model.evaluate(cached_test, return_dict=True)

"""Test set performance is much worse than training performance. This is due to two factors:

1. Our model is likely to perform better on the data that it has seen, simply because it can memorize it. This overfitting phenomenon is especially strong when models have many parameters. It can be mediated by model regularization and use of user and movie features that help the model generalize better to unseen data.
2. The model is re-recommending some of users' already watched movies. These known-positive watches can crowd out test movies out of top K recommendations.

The second phenomenon can be tackled by excluding previously seen movies from test recommendations. This approach is relatively common in the recommender systems literature, but we don't follow it in these tutorials. If not recommending past watches is important, we should expect appropriately specified models to learn this behaviour automatically from past user history and contextual information. Additionally, it is often appropriate to recommend the same item multiple times (say, an evergreen TV series or a regularly purchased item).

This layer will perform _approximate_ lookups: this makes retrieval slightly less accurate, but orders of magnitude faster on large candidate sets.
"""

# Export the query model.
model_base_dir = '/tmp'
path = os.path.join(model_base_dir, "model")

# Save the index.
tf.saved_model.save(
    lr_model,
    path)

# Load it back; can also be done in TensorFlow Serving.
loaded = tf.saved_model.load(path)

# Pass a user id in, get top predicted movie titles back.
# scores, titles = loaded({'user_id': np.array(['Wuming007 Wu']), 'course_id': np.array(['2356382'])})
lr_model({'user_id': np.array(['Wuming007 Wu']), 'course_id': np.array(['2356382'])})

# print(f"Recommendations: {titles[0][:3]}")
