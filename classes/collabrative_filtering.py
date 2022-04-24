import imp
import numpy as np
import pandas as pd
from classes.helper import get_retrieval_model, get_ranking_model, get_course



def collabrativefiltering_recommender(user_id):
    # Load it back; can also be done in TensorFlow Serving.
    retrieval_model_loaded = get_retrieval_model()
    ranking_model_loaded = get_ranking_model()

    # Pass a user id in, get top predicted movie titles back.
    scores, course_ids= retrieval_model_loaded([user_id])
    course_ids = [course_id.decode('utf-8') for course_id in course_ids[0].numpy()]
    dic1 = {'user_id': [user_id for i in range(len(course_ids))], 'course_id': course_ids}
    df = pd.DataFrame(dic1)
    df.drop_duplicates(subset=['course_id'], inplace=True)

    rating_list = df.to_dict('list')
    predict_rating = {}
    ranking_courses = ranking_model_loaded(rating_list).numpy()
    df['rating'] = ranking_courses
    df.sort_values(by='rating', ascending=False, ignore_index=True, inplace=True)
    df_courses = get_course()
    df_courses['id'] = df_courses.id.astype('str')
    df_courses.drop_duplicates(subset=['id'], inplace=True)
    print(df.course_id.to_numpy())

    output = df_courses[df_courses.id.isin(df.course_id.to_numpy())]
    # output = df_courses[np.in1d(df_courses.id.values, df.course_id.values)]
    print(output)
    if (len(output.index.values) < 10) and (len(output.index.values) >0):
        return output[['title', 'url']]
    else:
        return output[['title', 'url']]
