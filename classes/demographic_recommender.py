from classes.helper import get_user_by_id, age_categorization, is_student, cat_code_fetch, get_gender_model, get_age_category_model, get_student_model
from classes.clustering_test import kmeans_predict
import pickle
import os

base_dir = os.path.dirname(os.path.dirname(__file__))

def demographic_recommender_system(user_id):
    user_details = get_user_by_id(user_id)[0]
    gender=user_details['gender']
    dob = user_details['birthdate']
    age, age_category = age_categorization(dob)
    student = is_student(age)

    # print(age, age_category, gender, student)
    age_category_model = get_age_category_model()
    gender_model = get_gender_model()
    is_student_model = get_student_model()

    age_category_code = cat_code_fetch(age_category_model, age_category)
    gender_code = cat_code_fetch(gender_model, gender)
    student_code = cat_code_fetch(is_student_model, student)

    return kmeans_predict(gender_code, age_category_code, student_code)
    
    # print(age_category_model, gender_model, is_student_model)
    # print(age_category_code, gender_code, student_code)