from operator import index
from xml.sax import default_parser_list
import pandas as pd
import os
from datetime import datetime, date
import pickle
import tensorflow as tf

base_path = os.path.dirname(os.path.dirname(__file__))
def get_users():
    df_user = pd.read_csv(os.path.join(base_path, 'datasets/udemy_user_data.csv'))
    return df_user

def get_user_by_id(user_id):
    users = get_users()
    user_details = users.loc[users.name==user_id].to_dict('records')
    return user_details
    
def get_course():
    df_course = pd.read_csv(os.path.join(base_path, 'datasets/preprocessed_udemy_courses.csv'))    
    return df_course

def get_reviews():
    df_review = pd.read_csv(os.path.join(base_path, 'datasets/updated_rating_with_user_name.csv'))
    return df_review

def get_age_category_model():
    age_category_model = pickle.load(open(os.path.join(base_path,'models/age_category.pkl'), 'rb'))
    return age_category_model

def get_gender_model():
    gender_model = pickle.load(open(os.path.join(base_path,'models/gender.pkl'), 'rb'))
    return gender_model

def get_student_model():
    is_student_model = pickle.load(open(os.path.join(base_path,'models/is_student.pkl'), 'rb'))
    return is_student_model

def get_kmeans_labels():
    kmeans_labels = pickle.load(open(os.path.join(base_path,'models/kmeans_labels.pkl'), 'rb'))
    return kmeans_labels   

def get_kmeans_model():
    kmeans_model = pickle.load(open(os.path.join(base_path,'models/kmeans_model.pkl'), 'rb'))
    return kmeans_model  

def get_retrieval_model():
    retrieval_model_loaded = tf.saved_model.load(os.path.join(base_path, 'models/retrival_full_model'))
    return retrieval_model_loaded

def get_ranking_model():
    ranking_model_loaded = tf.saved_model.load(os.path.join(base_path, 'models/ranking_full_model'))
    return ranking_model_loaded    

def age_categorization(birthdate):
    born = datetime.strptime(birthdate, '%Y-%m-%d')
    today = date.today()
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    if age > 0 and age <= 16:
        category = 'child'
    elif age > 16 and age <= 30:
        category = 'young adults'
    elif age > 30 and age <= 45:
        category = 'middle-aged adults'
    elif age > 45:
        category = 'old-aged adults'      
    else:
        category = None
    return age, category

def is_student(age):
    if age > 5 and age <= 30:
        is_Student = 'yes'
    else:
        is_Student = 'no'
    return is_Student        

def cat_code_fetch(model, value):
    for k, v in model.items():
        if v==value:
            return k
