from classes.collabrative_filtering import collabrativefiltering_recommender
from classes.demographic_recommender import demographic_recommender_system
from classes.helper import get_reviews
import streamlit as st

reviews = get_reviews()

def recommend():
    st.title("Hybrid Recommender System")  # app name
    with st.form(key='user_id_form'):  # key of the form - emotion_clf_form
        user_id = st.text_input("Type your user id")  # text area for input data
        submit_text = st.form_submit_button(label='Submit')  # form submit button

        if submit_text:
            review_count = len(reviews[reviews.user_displayname==user_id].index.values)

            if review_count >= 2:
                df = collabrativefiltering_recommender(user_id)
                reason = 'User have favourable ratings'
                recommender = 'Collabrative Filtering'
            else:
                df = demographic_recommender_system(user_id)
                reason = "User don't have favourable ratings, so user is cold start user"    
                recommender = 'Demographic Based'

            hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
            # CSS to inject contained in a string
            hide_dataframe_row_index = """
                        <style>
                        .row_heading.level0 {display:none}
                        .blank {display:none}
                        </style>
                        """

            # Inject CSS with Markdown
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

            # Display a static table
            st.dataframe(df) 
            st.write(f"Recommender")
            st.success(f"{recommender}")
            st.write(f'Reason')
            st.info(f"{reason}") 
