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
# from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from PIL import Image

# Tableau
from streamlit_embedcode import github_gist
import streamlit.components.v1 as components


st.title('Educación a distancia en la época de la COVID-19')


st.sidebar.title("Panel de Navegación")
selection = st.sidebar.radio(
    "Go to",
    [
        'Introducción',
        'Estadisticas descriptivas',
        'Modelo',
        'Mapa de la estimación',
        'Simulación de una intervención',
        'Vulneravilidad COVID-19'
        # 'Hoja para pruebas',
        # 'Pandas Profiling in Streamlit'
    ]
    )

if (selection != 'Introducción') & (selection != 'Estadisticas descriptivas'):
    # Crear base de datos
    Data_Base = pd.read_csv(
        "https://raw.githubusercontent.com/IngFrustrado/AppDS4A/master/Data_Base_1419.csv",
        encoding='UTF-8'
        )


    Data_Base2 = Data_Base.copy()

    risk = st.sidebar.slider(
        label="Theshold Score",
        min_value=int(Data_Base2['PUNT_GLOBAL'].mean() - 2 * Data_Base2['PUNT_GLOBAL'].std()),
        max_value=int(Data_Base2['PUNT_GLOBAL'].mean() + 1 * Data_Base2['PUNT_GLOBAL'].std()),
        value= int(Data_Base2['PUNT_GLOBAL'].mean() - 1 * Data_Base2['PUNT_GLOBAL'].std()),
        step=1
    )
    if risk != int(Data_Base2['PUNT_GLOBAL'].mean() - 1 * Data_Base2['PUNT_GLOBAL'].std()):
        st.sidebar.warning(
            'Our analysis was conducted with a threshold score of ' +
            str(
                int(Data_Base2['PUNT_GLOBAL'].mean() - 1 * Data_Base2['PUNT_GLOBAL'].std())
            ))
        pass

    Data_Base2['Riesgo'] = 2
    Data_Base2['Riesgo'] = np.where(Data_Base2['PUNT_GLOBAL'] < risk, 1, 0)

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

    # # VOLVERLA CATEGORICA
    Test['Riesgo'] = np.where(Test['PUNT_GLOBAL'] < risk, 'Riesgo', 'No Riesgo')

    Test['riesgo_forest'] = rf_model.predict(Test[variables1])
    Test['riesgo_forest'] = np.where(Test.riesgo_forest < 0.5, 0, 1)
    Test['riesgo_regression'] = rf_modelD.predict(Test[variables1])
    # Test['riesgo_regression'] = np.where(Test.riesgo_regression < 0.5, 0, 1)
    Test['riesgo_logit'] = logit1_res.predict(Test[variables1])
    Test['riesgo_logit'] = np.where(Test.riesgo_logit < 0.5, 0, 1)

    Test['Riesgo_total'] = Test.riesgo_forest+Test.riesgo_logit+Test.riesgo_regression
    pass

Colores = ['#9BCCEA', '#7BA0BD', '#587C95', '#37546B']
# Graficos
if (selection == 'Introducción'):
    st.image(
        Image.open('img/Front.jpg'),
        # caption='Sunrise by the mountains',
        use_column_width=True
        )
    pass


if (selection == 'Estadisticas descriptivas'):
    # VariablesNum = [
    #     'PUNT_GLOBAL',
    #     'FAMI_TIENEINTERNET',
    #     'FAMI_TIENECOMPUTADOR',
    #     'ESTU_TIENEETNIA',
    #     'COLE_NATURALEZA',
    #     'ConexMilHab',
    #     'PoblacionTotal',
    #     'Indice_Rural'
    # ]

    # varx = st.sidebar.selectbox(
    #     label="Variable axis x",
    #     options=VariablesNum,
    #     index=0,
    # )

    # vary = st.sidebar.selectbox(
    #     label="Variable axis y",
    #     options=VariablesNum,
    #     index=0,
    # )

    # colorsList = matplotlib.colors.ListedColormap(
    #     ['#FFFFFF', '#6495ED', '#FFA500', '#FF4500'])

    # # HexBin =
    # plt.hexbin(
    #     Data_Base2[Data_Base2.Ano == Anno][varx],
    #     Data_Base2[Data_Base2.Ano == Anno][vary],
    #     gridsize=(30, 15),
    #     cmap=colorsList
    # )

    # st.pyplot()

    # st.video('https://www.youtube.com/watch?v=u6d9Eeg1jok')

    components.iframe(
        "https://public.tableau.com/views/Dashboard_Icfes_v2/Departamento?:showVizHome=no&:embed=true", scrolling=True, width = 1000, height=900)

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

    # fig2 = px.scatter(
    #     Test,
    #     x= varx,
    #     y= vary,
    #     title='Computer ownership vs Global Score',
    #     # labels={
    #     #     "FAMI_TIENECOMPUTADOR": "Computer ownership",
    #     #     "PUNT_GLOBAL": "Global Score"
    #     # },
    #     color='Intercepto',
    #     color_continuous_scale=['#FFA500', '#FFA500']
    # )
    # st.plotly_chart(fig2)


if (selection == 'Modelo'):
    Anno = st.selectbox(
        label="Año",
        options=[2014, 2015, 2016, 2017, 2018],
        index=0,
    )

    figHist = px.histogram(
        Data_Base2[Data_Base2.Ano == Anno],
        x="PUNT_GLOBAL",
        color="Riesgo",
        nbins=150
    )
    st.plotly_chart(figHist)

    fig = px.box(
        Test,
        x="Riesgo_total",
        y="ConexMilHab",
        color = "Riesgo"
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
if (selection == 'Mapa de la estimación'):
    file = "ShapeMap/MGN_MPIO_POLITICO.shp"
    MapaDpto = geopandas.read_file(file, encoding='utf-8')
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
        colors=Colores,
        index= [0, 1, 2, 3],
        vmin = min_cn,
        vmax = max_cn
    )

    tile = st.sidebar.selectbox(
        label="Mapa base",
        options=["CartoDB positron", "CartoDB dark_matter", "OpenStreetMap", "Stamen Toner", "Stamen Terrain", "Stamen Watercolor"],
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
            fields=['DPTO_CNMBR', 'MPIO_CNMBR', VariableGraph],
            aliases = ['Departamento', 'Municipio', 'Puntaje'], 
            localize = True
        )
    ).add_to(m_crime)

    colormap.add_to(m_crime)

    folium_static(m_crime)

if (selection == 'Simulación de una intervención'):
    opciones = ['Análisis País', 'Análisis Departamento', 'Análisis Municipio'] # una lista desplegable con los análisis
    analisis = st.sidebar.selectbox('Por favor seleccione un análisis de la lista.', opciones)
    #opciones = st.sidebar.radio("Seleccione el Analisis", ['Municipio', 'Departamento', 'Pais'])
    # Anàlisis por Departamento
    if analisis=='Análisis Departamento':
        #-------------------------------------------------------------------------------
        # Entradas de usuario ConexMilHab
        #-------------------------------------------------------------------------------
        Departamento_Sel = st.sidebar.selectbox(
            label="Filtro Departamento",
            options=Test['DEPARTAMENTO'].unique(),
            index=0,
        )
        ColumnsIn=['FAMI_TIENECOMPUTADOR','COLE_NATURALEZA','ConexMilHab']
        st.subheader("Análisis Departamento")
        #-------------------------------------------------------------------------------
        # Se calcula los valores base con los datos del usuario
        #-------------------------------------------------------------------------------
        Vector_Base_Departamento=Test[(Test['DEPARTAMENTO'] == Departamento_Sel) & (Test['Ano'] == max(Test['Ano']))]
        Risk_Base_Departamento=Vector_Base_Departamento['Riesgo_total'] 

        InD_FAMI_TIENECOMPUTADOR = st.slider(
            label="Porcentaje Familias con Computador",
            min_value=0,
            max_value=100,
            value= 0,
            step=1
        )
        InD_COLE_NATURALEZA = st.slider(
            label="Porcentaje de Colegios Privados",
            min_value=0,
            max_value=100,
            value= 0,
            step=1
        )
        InD_ConexMilHab = st.slider(
            label="Conexiones de Internet por Mil Habitantes",
            min_value=0,
            max_value=1000,
            value= 0,
            step=1
        )    
       
        #-------------------------------------------------------------------------------
        # Se calcula los vectores de predicción base con los datos del usuario
        #-------------------------------------------------------------------------------
        Vector_Usuario_Departamento = Vector_Base_Departamento.copy()
        Vector_Usuario_Departamento['FAMI_TIENECOMPUTADOR']=Vector_Usuario_Departamento['FAMI_TIENECOMPUTADOR']*(1+InD_FAMI_TIENECOMPUTADOR)
        Vector_Usuario_Departamento['COLE_NATURALEZA']=Vector_Usuario_Departamento['COLE_NATURALEZA']*(1+InD_FAMI_TIENECOMPUTADOR)
        Vector_Usuario_Departamento['ConexMilHab']=Vector_Usuario_Departamento['ConexMilHab']*(1+InD_FAMI_TIENECOMPUTADOR)
        #-------------------------------------------------------------------------------
        # Estimación
        #-------------------------------------------------------------------------------
        Estimate = Vector_Usuario_Departamento
        Estimate = Estimate[~Estimate.isin([np.nan, np.inf, -np.inf]).any(1)]

        Estimate['Intercepto'] = 1
        Estimate['riesgo_forest'] = rf_model.predict(Estimate[variables1])
        Estimate['riesgo_forest'] = np.where(Estimate.riesgo_forest < 0.5, 0, 1)
        Estimate['riesgo_regression'] = rf_modelD.predict(Estimate[variables1])
        Estimate['riesgo_logit'] = logit1_res.predict(Estimate[variables1])
        Estimate['riesgo_logit'] = np.where(Estimate.riesgo_logit < 0.5, 0, 1)
        Estimate['Riesgo_total'] = Estimate.riesgo_forest+Estimate.riesgo_logit+Estimate.riesgo_regression
        st.write(Estimate)
        st.dataframe(Estimate['Riesgo_total'])
        
    # Anàlisis por Municipio
    elif analisis=='Análisis Municipio':
        #-------------------------------------------------------------------------------
        # Entradas de usuario ConexMilHab
        #-------------------------------------------------------------------------------
        Departamento_Sel = st.sidebar.selectbox(
            label="Filtro Departamento",
            options=Test['DEPARTAMENTO'].unique(),
            index=0,
        )
        Municipio_Sel = st.sidebar.selectbox(
            label="Filtro Municipio",
            options=Test[Test['DEPARTAMENTO']==Departamento_Sel]['MUNICIPIO'].unique(),
            index=0,
        )
        ColumnsIn=['FAMI_TIENECOMPUTADOR','COLE_NATURALEZA','ConexMilHab']

        st.subheader("Análisis Municipio")
        #-------------------------------------------------------------------------------
        # Se calcula los valores base con los datos del usuario
        #-------------------------------------------------------------------------------
        Vector_Base_Municipio=Test[(Test['MUNICIPIO'] == Municipio_Sel) & (Test['Ano'] == max(Test['Ano']))]
        Risk_Base_Municipio=Vector_Base_Municipio['Riesgo_total'] 

        In_FAMI_TIENECOMPUTADOR = st.slider(
            label="Porcentaje Familias con Computador",
            min_value=0,
            max_value=100,
            value= int(Vector_Base_Municipio['FAMI_TIENECOMPUTADOR']*100),
            step=1
        )
        In_COLE_NATURALEZA = st.slider(
            label="Porcentaje de Colegios Privados",
            min_value=0,
            max_value=100,
            value= int(Vector_Base_Municipio['COLE_NATURALEZA']*100),
            step=1
        )
        In_ConexMilHab = st.slider(
            label="Conexiones de Internet por Mil Habitantes",
            min_value=0,
            max_value=1000,
            value= int(Vector_Base_Municipio['ConexMilHab']),
            step=1
        )       
        #-------------------------------------------------------------------------------
        # Se calcula los vectores de predicción base con los datos del usuario
        #-------------------------------------------------------------------------------
        Vector_Usuario_Municipio = Vector_Base_Municipio.copy()
        Vector_Usuario_Municipio[ColumnsIn]=[In_FAMI_TIENECOMPUTADOR/100,In_COLE_NATURALEZA/100,In_ConexMilHab]
        #-------------------------------------------------------------------------------
        # Estimación
        #-------------------------------------------------------------------------------
        Estimate = Vector_Usuario_Municipio
        Estimate = Estimate[~Estimate.isin([np.nan, np.inf, -np.inf]).any(1)]

        Estimate['Intercepto'] = 1
        Estimate['riesgo_forest'] = rf_model.predict(Estimate[variables1])
        Estimate['riesgo_forest'] = np.where(Estimate.riesgo_forest < 0.5, 0, 1)
        Estimate['riesgo_regression'] = rf_modelD.predict(Estimate[variables1])
        Estimate['riesgo_logit'] = logit1_res.predict(Estimate[variables1])
        Estimate['riesgo_logit'] = np.where(Estimate.riesgo_logit < 0.5, 0, 1)
        Estimate['Riesgo_total'] = Estimate.riesgo_forest+Estimate.riesgo_logit+Estimate.riesgo_regression
        #st.write(Estimate)
        st.dataframe(Estimate['Riesgo_total'])
        #-----------------------------------------------------------
        # Funciona con Plotly mapbox
        #-----------------------------------------------------------

        #df =  MapaDpto
        #import plotly.express as px

        #DirCol = pd.read_excel("DirecionesCol.xlsx")
        #DirCol['Magnitude'] = 10
        #MapaCol = px.scatter_mapbox(DirCol, lat='LATITUD', lon='LONGITUD', #color=Estimate['Riesgo_total'], size=Estimate['Riesgo_total'],
        #                  color_continuous_scale=px.colors.cmocean.tempo_r, size_max=24, zoom=4,
        #                  center= {"lat": 4.570868, "lon": -74.2973328},
        #                  mapbox_style="carto-positron")
        #st.plotly_chart(MapaCol)
        #st.write(DirCol)

        #--------------------
        file = "ShapeMap/MGN_MPIO_POLITICO.shp"
        MapaDpto = geopandas.read_file(file)
        MapaDpto['MPIO_CCDGO_C'] = pd.to_numeric(MapaDpto['DPTO_CCDGO'] + MapaDpto['MPIO_CCDGO'])
        #FiltroMunicipio=Estimate[Estimate['MUNICIPIO']==Municipio_Sel]['COLE_COD_MCPIO_UBICACION'].reset_index().stack()
        MapaDpto = MapaDpto.join(Estimate.set_index('COLE_COD_MCPIO_UBICACION'), how = 'left', on = 'MPIO_CCDGO_C')
        MapaDpto.fillna(0, inplace = True)
        #MapaDpto = MapaDpto[MapaDpto.MPIO_CCDGO_C == FiltroMunicipio[1]]
        DataFilter = pd.DataFrame(MapaDpto.drop(columns='geometry'))
        VariableGraph = 'Riesgo_total'
        colormap = branca.colormap.LinearColormap(
            colors = ['#FFFFFF', '#6495ED', '#FFA500', '#FF4500'],
            index= [0, 1, 2, 3],
            vmin = 0,
            vmax = 3
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
            name = 'Riesgo SABER PRO - Colombia',
            style_function = nombreestilo,
            tooltip = folium.GeoJsonTooltip(
                fields=['DPTO_CNMBR', 'MPIO_CNMBR', VariableGraph],
                aliases = ['Departamento', 'Municipio', 'Riesgo'], 
                localize = True
            )
        ).add_to(m_crime)

        colormap.add_to(m_crime)

        folium_static(m_crime)


# if selection == 'Hoja para pruebas':
#     DEPARTAMENTO_all = sorted(Data_Base.DEPARTAMENTO.unique().astype(str))
#     DEPARTAMENTO = st.sidebar.selectbox(
#         "Select DEPARTAMENTO",
#         ['All'] + DEPARTAMENTO_all
#     )

#     MUNICIPIO_all = sorted(
#         Data_Base[Data_Base.DEPARTAMENTO == DEPARTAMENTO].MUNICIPIO.unique())
#     MUNICIPIO = st.sidebar.selectbox(
#         "Select MUNICIPIO",
#         ['All'] + MUNICIPIO_all
#     )
#     if MUNICIPIO == 'All':
#         pass
#         if DEPARTAMENTO == 'All':
#             st.write(Data_Base)
#             pass
#         else:
#             st.write(Data_Base[(Data_Base.DEPARTAMENTO == DEPARTAMENTO)])
#         pass
#     else:
#         st.write(Data_Base[
#             (Data_Base.DEPARTAMENTO == DEPARTAMENTO)
#             & (Data_Base.MUNICIPIO == MUNICIPIO)
#         ])
#         pass
#     # COLE_COD_MCPIO_UBICACION_all = Data_Base.COLE_COD_MCPIO_UBICACION.unique()
#     # COLE_COD_MCPIO_UBICACION = st.selectbox(
#     #     "Select COLE_COD_MCPIO_UBICACION",
#     #     COLE_COD_MCPIO_UBICACION_all
#     # )
#     # st.write(Data_Base[(Data_Base.COLE_COD_MCPIO_UBICACION == COLE_COD_MCPIO_UBICACION)])
#     # st.write(Data_Base.columns)
#     # selected = st.selectbox('Select one option:', [
#     #                         '', 'First one', 'Second one'], format_func=lambda x: 'Select an option' if x == '' else x)
#     # if selected:
#     #     st.success('Yay! 🎉')
#     # else:
#     #     st.warning('No option is selected')
#     pass

if selection == 'Vulneravilidad COVID-19':
    url = "https://www.datos.gov.co/api/views/gt2j-8ykr/rows.csv?accessType=DOWNLOAD"
    DataCOVID = pd.read_csv(url)

    filtroNoEstado = ['Recuperado', 'Fallecido']
    DataCOVID  = DataCOVID[~DataCOVID['Estado'].isin(filtroNoEstado)]
    COVIDAgrup = DataCOVID.groupby(DataCOVID['Código DIVIPOLA']).count()['ID de caso'].reset_index()
    COVIDAgrup = COVIDAgrup.rename(columns={"ID de caso": "CasosActivos"})

    CovidTest = Test.merge(COVIDAgrup, how='left' ,left_on='COLE_COD_MCPIO_UBICACION', right_on='Código DIVIPOLA')
    CovidTest['CasosActivos'] = CovidTest['CasosActivos'].fillna(0)
    CovidTest=CovidTest.drop(columns=['Ano','PoblacionTotal','NoAccesosFijos','Indice_Rural','Código DIVIPOLA'])

    Poblacion_2020 = pd.read_excel("2020-poblacion.xlsx")#.dropna()
    covid_19=CovidTest.merge(Poblacion_2020[['Municipio','total','Rural']], how='left' ,left_on='COLE_COD_MCPIO_UBICACION', right_on='Municipio')
    covid_19['ContagioMilHab'] = 1000 * covid_19['CasosActivos'] / covid_19['total']
    covid_19['Indice_Rural']=covid_19['Rural']/covid_19['total'].round(2)

    Column = st.selectbox(
        label="Variable del eje y",
        options=["ConexMilHab", 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'ESTU_TIENEETNIA',
                 'COLE_NATURALEZA', 'PUNT_GLOBAL', 'ContagioMilHab', "Indice_Rural"],
        index=0,
    )
    Row = st.selectbox(
        label="Variable del eje x",
        options=["Indice_Rural", "ConexMilHab", 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'ESTU_TIENEETNIA', 'COLE_NATURALEZA', 'PUNT_GLOBAL', 'ContagioMilHab'],
        index=0,
    )
    Size = st.selectbox(
        label="Variable del tamaño",
        options=["ContagioMilHab", "ConexMilHab", 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'ESTU_TIENEETNIA', 'COLE_NATURALEZA', 'PUNT_GLOBAL', "Indice_Rural"],
        index=0,
    )

    if Row in ["ContagioMilHab", "ConexMilHab", 'PUNT_GLOBAL']:
        BoolX = True
        pass
    else:
        BoolX = False
        pass

    if Column in ["ContagioMilHab", "ConexMilHab", 'PUNT_GLOBAL']:
        BoolY = True
        pass
    else:
        BoolY = False
        pass

    fig = px.scatter(
        covid_19[covid_19.ContagioMilHab > 1],
        x=Row,
        y=Column,
        hover_name="MUNICIPIO",
        color="Riesgo_total",
        color_continuous_scale= Colores,
        size=Size,
        log_y=BoolY,
        log_x=BoolX,
        )
    
    st.plotly_chart(fig.update_traces(mode='markers', marker_line_width=1.5))

    # st.write(covid_19.columns)
    pass

