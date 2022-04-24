# -*- coding: utf-8 -*-
"""clustering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Khl00BImsc0XbMMyLwkq78DyGHzmXK-v
"""

import numpy as np
import pandas as pd
import pickle
from classes.helper import get_users, get_course, get_reviews, get_kmeans_model, get_kmeans_labels

kmeans = get_kmeans_model()
labels = get_kmeans_labels()

def kmeans_predict(gender_code, age_category_code, student_code):
# random predict variable
    predict_cluster = kmeans.predict([[gender_code, age_category_code, student_code]])
    predict_cluster

    df = get_users()
    indexes = np.where(labels==predict_cluster)[0]
    filter_df = df[df.index.isin(indexes)]

    """# found the top 10 courses in same user cluster"""

    df_courses = get_reviews()
    df_filter_course_reviews = df_courses[df_courses.user_displayname.isin(filter_df.name)]
    courses_with_average_rating = df_filter_course_reviews.groupby('course_id')['rating'].mean().sort_values(axis=0, ascending=False)
    courses_with_average_rating[:10]

    """# top 10 courses in tabular format"""

    df_courses = get_course()
    df_courses.drop_duplicates(subset=['id'], inplace=True)
    output = df_courses[df_courses.id.isin(courses_with_average_rating[:10].index)][['title', 'url']]
    return output