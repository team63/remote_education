import pandas as pd
import numpy as np

import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestRegressor

import sklearn
import matplotlib
import matplotlib.pyplot as plt

# Para Mapas
import plotly.express as px
import branca
import geopandas
import folium

# Streamlit
import streamlit as st
from streamlit_folium import folium_static
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from streamlit_embedcode import github_gist
import streamlit.components.v1 as components
from PIL import Image


st.title('Remote Education in the time of COVID-19')


st.sidebar.title("Navigation panel")
selection = st.sidebar.radio(
    "Go to",
    [
        'Home page',
        'Descriptive statistics',
        'Model',
        'Estimation (map)',
        'Simulation',
        'Hoja para pruebas',
        'Pandas Profiling in Streamlit'
    ]
    )

CarSD = st.sidebar.slider(
    label = "Desviaciones Estandar",
    min_value = 0.0,
    max_value = 3.0,
    value = 1.0,
    step = 0.05,
)
if CarSD != 1:
    st.warning('Nuestro analisis se realizo a una desviación estandar')
    pass

# Crear base de datos
Data_Base = pd.read_csv(
    "https://raw.githubusercontent.com/IngFrustrado/AppDS4A/master/Data_Base_1419.csv")


Data_Base2 = Data_Base.copy()

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
Test = Test[~Test.isin([np.nan, np.inf, -np.inf]).any(1)]

Test['Intercepto'] = 1
Test['riesgo_forest'] = rf_model.predict(Test[variables1])
Test['riesgo_forest'] = np.where(Test.riesgo_forest < 0.5, 0, 1)
Test['riesgo_regression'] = rf_modelD.predict(Test[variables1])
# Test['riesgo_regression'] = np.where(Test.riesgo_regression < 0.5, 0, 1)
Test['riesgo_logit'] = logit1_res.predict(Test[variables1])
Test['riesgo_logit'] = np.where(Test.riesgo_logit < 0.5, 0, 1)

Test['Riesgo_total'] = Test.riesgo_forest+Test.riesgo_logit+Test.riesgo_regression

# Graficos
if(selection == 'Home page'):
    st.image(
        Image.open('img/Front.jpg'),
        # caption='Sunrise by the mountains',
        use_column_width=True
        )
    pass


if(selection == 'Descriptive statistics'):
    Anno = st.sidebar.slider(
        label="Anno",
        min_value = 2014,
        max_value = 2018,
        value = 2018,
        step = 1
        )

    figHist = px.histogram(
        Data_Base2[Data_Base2.Ano == Anno],
        x="PUNT_GLOBAL",
        color="Riesgo",
        nbins=150
        )
    st.plotly_chart(figHist)

    VariablesNum = [
        'PUNT_GLOBAL',
        'FAMI_TIENEINTERNET',
        'FAMI_TIENECOMPUTADOR',
        'ESTU_TIENEETNIA',
        'COLE_NATURALEZA',
        'ConexMilHab',
        'PoblacionTotal',
        'Indice_Rural'
    ]

    varx = st.sidebar.selectbox(
        label="Variable axis x",
        options=VariablesNum,
        index=0,
    )

    vary = st.sidebar.selectbox(
        label="Variable axis y",
        options=VariablesNum,
        index=0,
    )

    colorsList = matplotlib.colors.ListedColormap(
        ['#FFFFFF', '#6495ED', '#FFA500', '#FF4500'])

    # HexBin =
    plt.hexbin(
        Data_Base2[Data_Base2.Ano == Anno][varx],
        Data_Base2[Data_Base2.Ano == Anno][vary],
        gridsize=(30, 15),
        cmap=colorsList
    )
    st.pyplot()


    # st.write(Test)
    # fig = px.scatter(
    #     Test,
    #     x="ConexMilHab",
    #     y="PUNT_GLOBAL",
    #     color="FAMI_TIENEINTERNET",
    #     size='PoblacionTotal',
    #     hover_data = ['MUNICIPIO']
    # )
    # st.plotly_chart(fig)

    fig2 = px.scatter(
        Test,
        x= varx,
        y= vary,
        title='Computer ownership vs Global Score',
        # labels={
        #     "FAMI_TIENECOMPUTADOR": "Computer ownership",
        #     "PUNT_GLOBAL": "Global Score"
        # },
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

    DPTO_CNMBR_all = sorted(MapaDpto.DPTO_CNMBR.unique().astype(str))
    DPTO_CNMBR = st.sidebar.selectbox(
        "Select DPTO_CNMBR",
        ['All'] + DPTO_CNMBR_all
    )
    if DPTO_CNMBR != 'All':
        MapaDpto = MapaDpto[MapaDpto.DPTO_CNMBR == DPTO_CNMBR]
        DataFilter = pd.DataFrame(MapaDpto.drop(columns='geometry'))
        st.write(
            px.pie(
                DataFilter[DataFilter.DPTO_CNMBR == DPTO_CNMBR],
                values='Intercepto',
                names='Riesgo_total'  # , title='Population of European continent'
            )
        )
        pass

    VariableGraph = 'Riesgo_total'

    min_cn, max_cn = MapaDpto[VariableGraph].quantile([0.01,0.99]).apply(round, 2)

    colormap = branca.colormap.LinearColormap(
        colors = ['#FFFFFF', '#6495ED', '#FFA500', '#FF4500'],
        index= [0, 1, 2, 3],
        vmin = min_cn,
        vmax = max_cn
    )

    tile = st.sidebar.selectbox(
        label="Mapa base",
        options=["CartoDB dark_matter", "OpenStreetMap", "Stamen Toner", "Stamen Terrain",
                "Stamen Watercolor", "CartoDB positron"],
        index=0,
    )

    m_crime = folium.Map(
        location=[4.570868, -74.2973328],
        zoom_start=5,
        tiles = tile
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


if selection == 'Hoja para pruebas':
    DEPARTAMENTO_all = sorted(Data_Base.DEPARTAMENTO.unique().astype(str))
    DEPARTAMENTO = st.sidebar.selectbox(
        "Select DEPARTAMENTO",
        ['All'] + DEPARTAMENTO_all
    )

    MUNICIPIO_all = sorted(
        Data_Base[Data_Base.DEPARTAMENTO == DEPARTAMENTO].MUNICIPIO.unique())
    MUNICIPIO = st.sidebar.selectbox(
        "Select MUNICIPIO",
        ['All'] + MUNICIPIO_all
    )
    if MUNICIPIO == 'All':
        pass
        if DEPARTAMENTO == 'All':
            st.write(Data_Base)
            pass
        else:
            st.write(Data_Base[(Data_Base.DEPARTAMENTO == DEPARTAMENTO)])
        pass
    else:
        st.write(Data_Base[
            (Data_Base.DEPARTAMENTO == DEPARTAMENTO)
            & (Data_Base.MUNICIPIO == MUNICIPIO)
        ])
        pass
    # COLE_COD_MCPIO_UBICACION_all = Data_Base.COLE_COD_MCPIO_UBICACION.unique()
    # COLE_COD_MCPIO_UBICACION = st.selectbox(
    #     "Select COLE_COD_MCPIO_UBICACION",
    #     COLE_COD_MCPIO_UBICACION_all
    # )
    # st.write(Data_Base[(Data_Base.COLE_COD_MCPIO_UBICACION == COLE_COD_MCPIO_UBICACION)])
    # st.write(Data_Base.columns)
    # selected = st.selectbox('Select one option:', [
    #                         '', 'First one', 'Second one'], format_func=lambda x: 'Select an option' if x == '' else x)
    # if selected:
    #     st.success('Yay! 🎉')
    # else:
    #     st.warning('No option is selected')
    pass

if selection == 'Pandas Profiling in Streamlit':
    pr = ProfileReport(Data_Base, explorative=True)
    st_profile_report(pr)
    pass

