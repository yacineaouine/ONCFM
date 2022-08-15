import streamlit as st

import numpy as np
import pandas as pd



def check_bill(diagonal, height_left, height_right, margin_low, margin_up, length):
    response = 0
    if response == 0:
        return st.write('Le billet est faux')
    else:
        return st.write('Le billet est vrai')

with st.container():
    st.image("./logo.png")
    st.title("Votre billet est-il faux ?")

response = "Entrer les valeurs ou télécharger un fichier"


#Upload et Formulaire
with st.sidebar:

    with st.container():
        uploaded_file = st.file_uploader("Choisir le fichier csv")
        if uploaded_file is not None:

            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe) 

    st.write("OU entrez les dimensions du billets")  

    with st.form("my_form"):
        diagonal = st.number_input('Diagonal')
        height_left = st.number_input('Height Left')
        height_right = st.number_input('Height Right')
        margin_low = st.number_input('Margin Low')
        margin_up = st.number_input('Margin Up')
        length = st.number_input('Length')

        # Every form must have a submit button.
        submitted = st.form_submit_button("Vérifier")
        if submitted:
            resp = check_bill(diagonal, height_left, height_right, margin_low, margin_up, length)
            response = resp


with st.container():
    st.write(response)




    
