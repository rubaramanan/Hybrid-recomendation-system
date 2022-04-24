from classes.collabrative_filtering import collabrativefiltering_recommender
from classes.demographic_recommender import demographic_recommender_system
from classes.helper import get_reviews
import streamlit as st
from scripts.recommender import recommend
 
        
if __name__ == '__main__':
    menu = ["Collabrative", "Demographic", "Sentiment Analysis"]  # navigation menu
    choice = st.sidebar.selectbox("Menu", menu)  # select menu - left side bar pane
    # with st.sidebar:
    
    #     choice = st.selectbox("Choose Model", ["Home", "Models", "Inference", "Project"],
    #                         icons=['house', 'sliders', 'person-bounding-box', 'kanban'],
    #                         menu_icon="app-indicator", default_index=0,
    #                         styles={
    #         "container": {"padding": "5!important", "background-color": "#f0f2f6"},
    #         "icon": {"color": "orange", "font-size": "28px"}, 
    #         "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    #         "nav-link-selected": {"background-color": "#2C3845"},
    #     }
    #     )
    if choice == "Collabrative":
        recommend()
    # elif choice == "Demographic":
    #     demographic_recommender_system(user_id)        
