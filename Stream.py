import pandas as pd
import numpy as np

import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestRegressor

import sklearn
import matplotlib.pyplot as plt


import streamlit as st
import plotly.express as px

st.title('Vulnerabilidad de municipios en epoca de COVID')

# Crear base de datos
Data_Base = pd.read_csv("Data_Base_1419.csv")
Data_Base2 = Data_Base.copy()

CarSD = st.sidebar.slider(
    label="Desviaciones Estandar",
    min_value=0.1,
    max_value=3.0,
    value=0.0,
    step=0.1,
)

tile = st.sidebar.selectbox(
    label="Mapa base",
    options=["OpenStreetMap", "Stamen Toner", "Stamen Terrain",
    "Stamen Watercolor", "CartoDB positron", "CartoDB dark_matter"],
    index=0,
)

Data_Base2['Riesgo'] = 2
for i in [2014, 2015, 2016, 2017, 2018, 2019]:
    risk = Data_Base2[Data_Base2.Ano == i]['PUNT_GLOBAL'].mean() - CarSD * Data_Base2[Data_Base2.Ano == i]['PUNT_GLOBAL'].std()
    Data_Base2['Riesgo'] = np.where(i == Data_Base2.Ano, np.where(Data_Base2['PUNT_GLOBAL'] < risk, 1, 0), Data_Base2.Riesgo)

Data_Base1 = Data_Base2[Data_Base2.Ano < 2019]
Data_Base1 = Data_Base1[~Data_Base1.isin([np.nan, np.inf, -np.inf]).any(1)]

# Modelo LOGIT
Data_Base1['Intercepto'] = 1
variables = [
    'Intercepto',
    'FAMI_TIENEINTERNET', 
    'FAMI_TIENECOMPUTADOR',
    'ESTU_TIENEETNIA',
    'COLE_NATURALEZA',
    'ConexMilHab',
    'PoblacionTotal',
    'Indice_Rural'
    ]

logit1 = sm.Logit(Data_Base1['Riesgo'], Data_Base1[variables])
logit1_res = logit1.fit()

var_tmp = variables
while 0 != sum(logit1_res.pvalues > 0.05):
    T = len(logit1_res.pvalues)
    #print(logit1_res.pvalues.sort_values(ascending=True).reset_index()[-1:]['index'])
    var_tmp = logit1_res.pvalues.sort_values(
        ascending=True).reset_index()[:(T-1)]['index']
    logit1 = sm.Logit(Data_Base1['Riesgo'], Data_Base1[var_tmp])
    logit1_res = logit1.fit()

variables1 = var_tmp
pscore = logit1_res.predict(Data_Base1[variables1])
Data_Base1['pscore'] = pscore

# REGRESSION TREE
clf = DecisionTreeRegressor(max_depth=3)
clf = clf.fit(Data_Base1[variables1], Data_Base1['Riesgo'])
pscore_tree = clf.predict(Data_Base1[variables1])
Data_Base1['pscore_tree'] = pscore_tree

#RANDOM FOREST
rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
rf_model.fit(Data_Base1[variables1], Data_Base1['Riesgo'])
pscore_forest = rf_model.predict(Data_Base1[variables1])
Data_Base1['pscore_forest'] = pscore_forest

# Estimación
Test = Data_Base[Data_Base.Ano == 2019]
Test['Intercepto'] = 1
Test['riesgo_forest'] = rf_model.predict(Test[variables1])
Test['riesgo_forest'] = np.where(Test.riesgo_forest < 0.5, 0, 1)
Test['riesgo_regression'] = clf.predict(Test[variables1])
Test['riesgo_regression'] = np.where(Test.riesgo_regression < 0.5, 0, 1)
Test['riesgo_logit'] = logit1_res.predict(Test[variables1])
Test['riesgo_logit'] = np.where(Test.riesgo_logit < 0.5, 0, 1)

Test['Riesgo_total'] = Test.riesgo_forest+Test.riesgo_logit+Test.riesgo_regression

# Graficos
# st.write(Test)


fig = px.box(
    Test,
    x="Riesgo_total",
    y="ConexMilHab",
    # title='Connectivity vs Year', labels={
    # "Ano": "Year",
    # "ConexMilHab": "Connectivity"}
)
st.plotly_chart(fig)
