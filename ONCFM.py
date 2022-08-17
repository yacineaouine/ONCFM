import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor 
import statsmodels

from scipy.stats import t, shapiro

from collections import Counter


from sklearn import linear_model
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score)


#Import des fichiers CSV
data_billets = pd.read_csv("./SOURCE/billets.csv", sep = ';')

#Régression linéaire multiple pour déterminr les valeurs manquantes 'margin_low'
def prediction(data):
    data_billets["is_genuine_n"] = np.where(data_billets['is_genuine'] == True, 1, 0)
    data_dropna = data_billets.dropna()
    #Valeurs de margin_low à prévoir
    na_data = data_billets.dropna()
    only_na = data_billets[~data_billets.index.isin(na_data.index)]

    reg_multi = smf.ols('margin_low~diagonal+height_left+height_right+margin_up+length+is_genuine_n', data=data_dropna).fit()
    reg_multi = smf.ols('margin_low~diagonal+height_left+height_right+margin_up+is_genuine_n', data=data_dropna).fit()
    reg_multi = smf.ols('margin_low~height_left+height_right+margin_up+is_genuine_n', data=data_dropna).fit()
    reg_multi = smf.ols('margin_low~height_left+margin_up+is_genuine_n', data=data_dropna).fit()
    reg_multi = smf.ols('margin_low~margin_up+is_genuine_n', data=data_dropna).fit()

    only_na = only_na.fillna(0)
    a_prevoir = pd.DataFrame(only_na)
    margin_low_prev = reg_multi.predict(a_prevoir)
    margin_low_prev=margin_low_prev.to_frame()

    margin_low_prev=margin_low_prev.rename(columns={0: "margin_low_prev"})

    data_predicted=only_na.join(margin_low_prev)
    data_predicted=data_predicted.drop(columns=['margin_low'])
    data_predicted=data_predicted.rename(columns={"margin_low_prev": "margin_low"})
    data_final=pd.concat([data_dropna, data_predicted])

    data_final=data_final.drop(columns="is_genuine")
    list_col = ["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]
    xtrain, xtest, ytrain, ytest = train_test_split(data_final[list_col], data_final["is_genuine_n"], train_size=0.75)
    data_train = xtrain.join(ytrain)
    data_test = xtest.join(ytest)

    #instanciation du modèle
    modele_regLog = linear_model.LogisticRegression(random_state = 0,
    solver = 'liblinear', multi_class = 'auto')

    #training
    modele_regLog.fit(xtrain,ytrain)

    #précision du modèle
    precision = modele_regLog.score(xtest,ytest)

    #Predition
    prediction=modele_regLog.predict(xtest)

    # performing predictions on the train datdaset
    yhat = modele_regLog.predict(data) 
    prediction = list(map(round, yhat))


    return prediction, precision


def check_result(result):
    if result == 0 or result == 1:
        
        if result == 0:
            return 'Le billet est faux'
        else:
            return 'Le billet est vrai'
    else: "Un problème est survenu, veuillez réessayer"

if 'response' not in st.session_state:
    st.session_state['response'] = 'Entrer les valeurs ou télécharger un fichier'

if 'precision_modele' not in st.session_state:
    st.session_state['precision_modele'] = ''

#LOGO & TITRE
with st.container():
    st.image("./SOURCE/logo.png")
    st.title("Votre billet est-il faux ?")

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
            values = [diagonal, height_left, height_right, margin_low, margin_up, length]
            data = [values]
            result_prediction = prediction(data)[0][0]
            precision_modele = prediction(data)[1]

            result = check_result(result_prediction)
            st.session_state['response'] = result
            st.session_state['precision_modele'] = precision_modele
            

with st.container():
    st.write(st.session_state['response'])
    st.write(st.session_state['precision_modele'])
