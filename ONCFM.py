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
from sklearn.model_selection import train_test_split


#Import des fichiers CSV
data_billets = pd.read_csv("./SOURCE/billets.csv", sep = None, iterator = False)

#Régression linéaire multiple pour déterminr les valeurs manquantes 'margin_low'
def model_prediction(data):
    data_billets["is_genuine_n"] = np.where(data_billets['is_genuine'] == True, 1, 0)
    data_dropna = data_billets.dropna()
    #Valeurs de margin_low à prévoir
    na_data = data_billets.dropna()
    only_na = data_billets[~data_billets.index.isin(na_data.index)]

    #Backward Méthode
    reg_multi = smf.ols('margin_low~diagonal+height_left+height_right+margin_up+length', data=data_dropna).fit()

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

    #Instanciation du modèle
    modele_regLog = linear_model.LogisticRegression(random_state = 0, solver = 'liblinear', multi_class = 'auto')

    #Training
    modele_regLog.fit(xtrain,ytrain)

    #Précision du modèle
    precision = modele_regLog.score(xtest,ytest)

    #Prediction
    prediction=modele_regLog.predict(xtest)

    #Performing predictions on the train datdaset
    yhat = modele_regLog.predict(data) 
    prediction = list(map(round, yhat))

    return prediction, precision


#VARIABLES
if 'response' not in st.session_state:
    st.session_state['response'] = 'Entrer les valeurs ou télécharger un fichier'

if 'precision_modele' not in st.session_state:
    st.session_state['precision_modele'] = ''

uploaded_file=None
submitted=False
dataframe=None

#SIDEBAR
with st.sidebar:

    #LOGO LBT
    with st.container():
        st.image("./SOURCE/logo_lbt.jpg", width=100)

    #SEPARATEUR
    html_string = "<hr></hr>"
    st.markdown(html_string, unsafe_allow_html=True) 


    #UPLOAD
    st.title("Importez votre fichier")
    with st.container():
        uploaded_file = st.file_uploader("Choisir le fichier csv")
        if uploaded_file is not None:

            # On récupère le fichier
            dataframe = pd.read_csv(uploaded_file, sep = None, iterator = False)

            dataframe=dataframe.dropna()

            #on affiche l'aperçu du fichier
            st.write(dataframe)

            #On récupère les valeurs 
            list_col = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length'] 
            dataframe_val = dataframe[list_col]
            arr_result = model_prediction(dataframe_val.values)[0]

            #On récupère la valeur obtenue de la précision du modèle 
            precision_modele = model_prediction(dataframe_val.values)[1]
            st.session_state['precision_modele'] = str(round(precision_modele, 3)*100) + '%'


            #on remplace les valeurs 0/1 par faux/vrai
            arr_result_string = []
            for result in arr_result:
                if(result == 0): 
                    result = "FAUX"
                    arr_result_string.append(result)
                else: 
                    result = "VRAI"
                    arr_result_string.append(result)

            #On crée le dataframe avec les résultats
            dataframe["resultat"]=arr_result_string

            if 'id' in dataframe.columns:             
                dataframe_result = dataframe[["id", "resultat"]]
            else:
                dataframe_result = dataframe[["resultat"]]
      
    #SEPARATEUR
    html_string = "<hr></hr>"
    st.markdown(html_string, unsafe_allow_html=True)   

    st.markdown("""<p style="text-align:center;font-weight:bold">OU</p>""", unsafe_allow_html=True)  

    #SEPARATEUR
    html_string = "<hr></hr>"
    st.markdown(html_string, unsafe_allow_html=True)

    #FORMULAIRE
    st.title("Entrez les dimensions du billets")

    with st.form("my_form"):
        diagonal = st.number_input('Diagonal')
        height_left = st.number_input('Height Left')
        height_right = st.number_input('Height Right')
        margin_low = st.number_input('Margin Low')
        margin_up = st.number_input('Margin Up')
        length = st.number_input('Length')

        st.session_state['response'] = ''

        # Every form must have a submit button.
        submitted = st.form_submit_button("Vérifier")
        if submitted:
            values = [diagonal, height_left, height_right, margin_low, margin_up, length]
            data = [values]
            result_prediction = model_prediction(data)[0][0]
            precision_modele = model_prediction(data)[1]

            #On vérifie la valeur obtenue du résultat 0 ou 1 pour afficher le message
            if result_prediction == 0 or result_prediction == 1:
        
                if result_prediction == 0:
                    st.session_state['response'] = 'Le billet semble faux'
                else:
                    st.session_state['response'] = 'Le billet semble vrai'
            else: st.error('Un problème est survenu, veuillez réessayer', icon="🚨")

            #On récupère la valeur obtenue de la précision du modèle 
            st.session_state['precision_modele'] = str(round(precision_modele, 3)*100) + '%'


    #SEPARATEUR
    html_string = "<hr></hr>"
    st.markdown(html_string, unsafe_allow_html=True) 

    #FOOTER
    footer="""
    <style>
    a:link , a:visited{
        color: blue;
        background-color: transparent;
        text-decoration: underline;
    }

    a:hover,  a:active {
        color: red;
        background-color: transparent;
        text-decoration: underline;
    }

    .footer {
        margin-bottom: 35px;
        width: 100%;
        color: black;
        text-align: center;
    }
    </style>
    <div class="footer">
    <p>
    Developed with ❤ by 
    <a style='display: block; text-align: center;' href="https://yacineaouine-cv.littleboytechnology.fr/" target="_blank">Yacine Aouine</a>
    <a style='display: block; text-align: center;' href="https://littleboytechnology.fr/" target="_blank">Little Boy Technology</a>
    v1.3
    </p>
    </div>
    """
    st.markdown(footer,unsafe_allow_html=True)



#CONTAINER: LOGO & TITRE
with st.container():
    st.image("./SOURCE/logo.png")
    st.title("Votre billet est-il faux ?")


#SEPARATEUR
html_string = "<hr></hr>"
st.markdown(html_string, unsafe_allow_html=True)

#CONTAINER: RESULTAT
st.title('Votre résultat:')
with st.container():
    
    if(st.session_state['response'] == 'Le billet semble vrai' and submitted):
        st.balloons()
        st.success('Le billet semble vrai', icon="✅")
        st.info('Précision du modèle: ' + st.session_state['precision_modele'], icon="ℹ️")

    elif(st.session_state['response'] == 'Le billet semble faux' and submitted):
        st.snow()
        st.error('Le billet semble faux', icon="🚨")
        st.info('Précision du modèle: ' + st.session_state['precision_modele'], icon="ℹ️")
    
    elif(not submitted and uploaded_file is not None):
        dataframe_result
        st.info('Précision du modèle: ' + st.session_state['precision_modele'], icon="ℹ️")

    else:
        st.info('Entrer les valeurs ou télécharger un fichier', icon="ℹ️")

#SEPARATEUR
html_string = "<hr></hr>"
st.markdown(html_string, unsafe_allow_html=True)


#CONTAINER: DONNEES D'ENTRAINEMENTS
with st.expander("Voir les données d'entrainements"):
        st.write("""Les données suivantes ont permis d'entrainer l'algorithme:""") 
        st.write("""- 1 signifie que le BILLET EST VRAI""")
        st.write("""- O signifie que le BILLET EST FAUX""")
        data_billets



