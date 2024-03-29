# -*- coding: utf-8 -*-
"""basic_retrieval_course_rec_train_without_evaluate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aIYk36hCTJ9Au1J7jdOZNZLQyvnW9d8v
"""


import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

"""## Preparing the dataset"""

ratings_df = pd.read_csv('/content/drive/MyDrive/FYP-CRSRider/data/ramanan/filtered_review.csv')
ratings_df = ratings_df[['course_id','rating','user_displayname']]
favourable_rating_users = ratings_df['user_displayname'].value_counts().loc[lambda x: x>=2].loc[lambda x: x<100].index.values
ratings_df = ratings_df[ratings_df.user_displayname.isin(favourable_rating_users)]
ratings_df.course_id = ratings_df.course_id.astype('str')
courses_df = pd.read_csv('/content/drive/MyDrive/FYP-CRSRider/data/ramanan/preprocessed_udemy_courses.csv')
courses_df.course_id = courses_df.course_id.astype('str')

ratings = tf.data.Dataset.from_tensor_slices(dict(ratings_df))
courses = tf.data.Dataset.from_tensor_slices(dict(courses_df))
ratings_df.loc[ratings_df.course_id.isna()]
ratings_df.shape

for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)

for x in courses.take(1).as_numpy_iterator():
  pprint.pprint(x)

ratings = ratings.map(lambda x: {
    "course_id": x["course_id"],
    "user_id": x["user_displayname"],
})
courses = courses.map(lambda x: x["course_id"])

course_ids = courses.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_courses = np.unique(np.concatenate(list(course_ids)))
unique_users = np.unique(np.concatenate(list(user_ids)))

# unique_user_ids[:10]

"""## Implementing a model

Choosing the architecture of our model is a key part of modelling.

Because we are building a two-tower retrieval model, we can build each tower separately and then combine them in the final model.

### The query tower

Let's start with the query tower.

The first step is to decide on the dimensionality of the query and candidate representations:
"""

embedding_dimension = 32

"""A simple model like this corresponds exactly to a classic [matrix factorization](https://ieeexplore.ieee.org/abstract/document/4781121) approach. While defining a subclass of `tf.keras.Model` for this simple model might be overkill, we can easily extend it to an arbitrarily complex model using standard Keras components, as long as we return an `embedding_dimension`-wide output at the end.

### The candidate tower

We can do the same with the candidate tower.

### The full model

We can now put it all together into a model. TFRS exposes a base model class (`tfrs.models.Model`) which streamlines building models: all we need to do is to set up the components in the `__init__` method, and implement the `compute_loss` method, taking in the raw features and returning a loss value.

The base model will then take care of creating the appropriate training loop to fit our model.
"""

class CourseRetreival(tfrs.Model):
    def __init__(self):
        super().__init__()
        
        embedding_dims = 32
        self.user_model =  tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary= unique_users, mask_token=None),
            tf.keras.layers.Embedding(len(unique_users)+1, embedding_dims)
        ])

        self.course_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_courses, mask_token=None),
            tf.keras.layers.Embedding(len(unique_courses)+1, embedding_dims)
        ])

        self.task = tfrs.tasks.Retrieval(
                        metrics=tfrs.metrics.FactorizedTopK(
                        candidates=courses.batch(128).cache().map(self.course_model)
                        ))
        
    
    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features['user_id'])
        course_embeddings = self.course_model(features['course_id'])
        return self.task(user_embeddings, course_embeddings)

"""## Fitting and evaluating

After defining the model, we can use standard Keras fitting and evaluation routines to fit and evaluate the model.

Let's first instantiate the model.
"""

model = CourseRetreival()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

"""Then shuffle, batch, and cache the training and evaluation data.

Then train the  model:
"""

history = model.fit(ratings.batch(8192), epochs=3)

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model, k=20)
# recommends movies out of the entire movies dataset.
index.index_from_dataset(
  tf.data.Dataset.zip((courses.batch(100), courses.batch(100).map(model.course_model)))
)

"""This layer will perform _approximate_ lookups: this makes retrieval slightly less accurate, but orders of magnitude faster on large candidate sets."""

# Get recommendations.
_, out_course_ids = index(tf.constant(["Wuming007 Wu"]))
print(f"Recommendations for user Wuming007 Wu: {out_course_ids[0]}")

# Export the query model.
model_base_dir = '/content/drive/MyDrive/FYP-CRSRider/model/ramanan'
path = os.path.join(model_base_dir, "retrival_full_model")

# Save the index.
tf.saved_model.save(
    index,
    path
)

# Load it back; can also be done in TensorFlow Serving.
loaded = tf.saved_model.load(path)

# Pass a user id in, get top predicted movie titles back.
scores, titles = loaded(["42"])

print(f"Recommendations: {titles[0][:3]}")
