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

# Para Mapas
import folium
from streamlit_folium import folium_static
import branca
import geopandas

st.title('Vulnerabilidad de municipios en epoca de COVID')


st.sidebar.title("Navigation panel")
selection = st.sidebar.radio(
    "Go to",
    [
        'Home page',
        'Descriptive statistics',
        'Model',
        'Estimation (map)',
        'Simulation'
    ]
    )

# Crear base de datos
Data_Base = pd.read_csv("Data_Base_1419.csv")
Data_Base2 = Data_Base.copy()

CarSD = 1.5 
# st.sidebar.slider(
#     label="Desviaciones Estandar",
#     min_value=0.1,
#     max_value=3.0,
#     value=0.0,
#     step=0.1,
# )

# tile = st.sidebar.selectbox(
#     label="Mapa base",
#     options=["OpenStreetMap", "Stamen Toner", "Stamen Terrain",
#     "Stamen Watercolor", "CartoDB positron", "CartoDB dark_matter"],
#     index=0,
# )

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
rf_modelD = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_modelD.fit(Data_Base1[variables1], Data_Base1['Riesgo'])
pscore_forestd = rf_modelD.predict(Data_Base1[variables1])
Data_Base1['pscore_forestd'] = pscore_forestd

# RANDOM FOREST
rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
rf_model.fit(Data_Base1[variables1], Data_Base1['Riesgo'])
pscore_forest = rf_model.predict(Data_Base1[variables1])
Data_Base1['pscore_forest'] = pscore_forest

# Estimación
Test = Data_Base[Data_Base.Ano == 2019]
Test['Intercepto'] = 1
Test['riesgo_forest'] = rf_model.predict(Test[variables1])
Test['riesgo_forest'] = np.where(Test.riesgo_forest < 0.5, 0, 1)
Test['riesgo_regression'] = rf_modelD.predict(Test[variables1])
# Test['riesgo_regression'] = np.where(Test.riesgo_regression < 0.5, 0, 1)
Test['riesgo_logit'] = logit1_res.predict(Test[variables1])
Test['riesgo_logit'] = np.where(Test.riesgo_logit < 0.5, 0, 1)

Test['Riesgo_total'] = Test.riesgo_forest+Test.riesgo_logit+Test.riesgo_regression

# Graficos
# st.write(Test)
if(selection == 'Descriptive statistics'):
    # st.write(Test)
    fig = px.scatter(
        Test,
        x="ConexMilHab",
        y="PUNT_GLOBAL",
        color="FAMI_TIENEINTERNET",
        size='PoblacionTotal',
        hover_data = ['MUNICIPIO']
    )
    st.plotly_chart(fig)

    fig2 = px.scatter(
        Test,
        x="FAMI_TIENECOMPUTADOR",
        y="PUNT_GLOBAL",
        title='Computer ownership vs Global Score',
        labels={
            "FAMI_TIENECOMPUTADOR": "Computer ownership",
            "PUNT_GLOBAL": "Global Score"
        },
        color='Intercepto',
        color_continuous_scale=['#FFA500', '#FFA500']
    )
    st.plotly_chart(fig2)


if(selection == 'Model'):
    fig = px.box(
        Test,
        x="Riesgo_total",
        y="ConexMilHab",
        # title='Connectivity vs Year', labels={
        # "Ano": "Year",
        # "ConexMilHab": "Connectivity"}
    )
    
    st.plotly_chart(fig)

# # Mapa 1
# DirCol = pd.read_excel(
#     "DirecionesCol.xlsx")
# DirCol['Magnitude'] = 10

# MapaCol = px.scatter_mapbox(
#     DirCol,
#     lat='LATITUD', lon='LONGITUD',
#     # z = 'Magnitude',
#     # radius = 10,
#     center=dict(lat=3.76, lon=-76.52), zoom=5,
#     # mapbox_style = "stamen-terrain"
#     hover_name="COD_INST", hover_data=["NOM_INST"]
#     )

# MapaCol.update_layout(mapbox_style="open-street-map")
# MapaCol.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

# st.plotly((MapaCol))

# Mapa 2
if(selection == 'Estimation (map)'):
    file = "ShapeMap/MGN_MPIO_POLITICO.shp"
    MapaDpto = geopandas.read_file(file)
    MapaDpto['MPIO_CCDGO_C'] = pd.to_numeric(MapaDpto['DPTO_CCDGO'] + MapaDpto['MPIO_CCDGO'])

    MapaDpto = MapaDpto.join(Test.set_index('COLE_COD_MCPIO_UBICACION'), how = 'left', on = 'MPIO_CCDGO_C')
    MapaDpto.fillna(0, inplace = True)

    VariableGraph = 'Riesgo_total'

    min_cn, max_cn = MapaDpto[VariableGraph].quantile([0.01,0.99]).apply(round, 2)

    colormap = branca.colormap.LinearColormap(
        colors=['#FFFFFF', '#6495ED', '#FFA500', '#FF4500'],
        index= [0, 1, 2, 3],
        vmin = min_cn,
        vmax = max_cn
    )

    m_crime = folium.Map(
        location=[4.570868, -74.2973328],
        zoom_start=5,
        tiles="OpenStreetMap"
        )

    nombreestilo = lambda x: {
        'fillColor': colormap(x['properties'][VariableGraph]),
        'color': 'black',
        'weight':0,
        'fillOpacity':0.75
    }

    stategeo = folium.GeoJson(
        MapaDpto.to_json(),
        name = 'SABER PRO - Colombia',
        style_function = nombreestilo,
        tooltip = folium.GeoJsonTooltip(
            fields = ['MPIO_CNMBR', VariableGraph],
            aliases = ['Municipio', 'Puntaje'], 
            localize = True
        )
    ).add_to(m_crime)

    colormap.add_to(m_crime)

    folium_static(m_crime)

if(selection == 'Simulation'):
    DepartamentoFilter = st.sidebar.selectbox(
        label="Filtro Departamento",
        options=Test['DEPARTAMENTO'].unique(),
        index=0,
    )

    st.write(Test[Test['DEPARTAMENTO'] == DepartamentoFilter][['MUNICIPIO', 'DEPARTAMENTO', 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'ESTU_TIENEETNIA', 'COLE_NATURALEZA', 'PUNT_GLOBAL', 'PoblacionTotal', 'ConexMilHab', 'NoAccesosFijos', 'Indice_Rural']])
