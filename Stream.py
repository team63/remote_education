import pandas as pd
import numpy as np
import math
from scipy import stats

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


st.title('Educación a distancia en los tiempos del COVID-19')


st.sidebar.title("Panel de Navegación")
selection = st.sidebar.radio(
    "Ir a",
    [
        'Introducción',
        'Estadisticas descriptivas',
        'Modelo',
        'Mapa de la estimación',
        'Simulación de una intervención',
        'Vulneravilidad COVID-19',
        'Conclusiones',
        'Modelo dinámico (Avanzado)'
        # 'Hoja para pruebas',
        # 'Pandas Profiling in Streamlit'
    ]
    )
# Colores = ['#FF4500', '#D75043', '#AF5A86', '#8765C9'] # Azules
# Colores = ['#FF4500', '#D75043', '#AF5A86', '#8765C9']  # Spider-Man
# Colores = ['#FFA500', '#D79043', '#AF7A86', '#8765C9'] # Amarillo-Morado
# Colores = ['#FF4500', '#E65F55', '#CC7AAA', '#B394FF']
# Colores = ['#B394FF', '#CC7AAA', '#E65F55', '#FF4500']


if (selection != 'Introducción') & (selection != 'Estadisticas descriptivas') & (selection != 'Modelo dinámico (Avanzado)'):
    # Crear base de datos
    Data_Base = pd.read_csv(
        "https://raw.githubusercontent.com/IngFrustrado/AppDS4A/master/Data_Base_1419.csv",
        encoding='UTF-8'
        )

    UmbralDefault = math.ceil(Data_Base[Data_Base.Ano == 2019]['PUNT_GLOBAL'].mean() - 1 * Data_Base[Data_Base.Ano == 2019]['PUNT_GLOBAL'].std())
    
    risk = 213

    QuanUmbral = stats.percentileofscore(Data_Base[Data_Base.Ano == 2019]['PUNT_GLOBAL'], risk) / 100

    Data_Base['Riesgo'] = 2
    for i in [2014, 2015, 2016, 2017, 2018, 2019]:
        riskFor = Data_Base[Data_Base.Ano == i]['PUNT_GLOBAL'].quantile(QuanUmbral)
        Data_Base['Riesgo'] = np.where(i == Data_Base.Ano, np.where(Data_Base['PUNT_GLOBAL'] < riskFor, 1, 0), Data_Base.Riesgo)

    if risk != math.ceil(UmbralDefault):
        st.sidebar.warning(
            'Nuestro análisis se llevó a cabo con una puntaje umbral de ' +
        str(
            math.ceil(UmbralDefault)
        ))
        pass

    Data_Base2 = Data_Base.copy()
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
    Test['riesgo_forest'] = rf_model.predict(Test[variables1])
    Test['riesgo_forest'] = np.where(Test.riesgo_forest < 0.5, 0, 1)
    Test['riesgo_regression'] = rf_modelD.predict(Test[variables1])
    # Test['riesgo_regression'] = np.where(Test.riesgo_regression < 0.5, 0, 1)
    Test['riesgo_logit'] = logit1_res.predict(Test[variables1])
    Test['riesgo_logit'] = np.where(Test.riesgo_logit < 0.5, 0, 1)

    Test['Riesgo_total'] = Test.riesgo_forest+Test.riesgo_logit+Test.riesgo_regression
    pass

if (selection == 'Introducción'):
    st.image(
        Image.open('img/Front.jpg'),
        # caption='Sunrise by the mountains',
        use_column_width=True
    )
    st.markdown("""<p><span style="font-size:10px;text-align: justify;color: #808080;">Foto tomada de: <span style="text-decoration: underline;"><a style="color: #808080; text-decoration: underline;" href="https://www.mineducacion.gov.co/portal/Preescolar-basica-y-media/">MinEducaci&oacute;n</a></span>.</span></p>""",
    unsafe_allow_html=True)

    st.markdown("""<h1>Aquí encontraras...</h1> <p style="text-align: justify">Con la reciente aparici&oacute;n del COVID-19, se hizo necesario implementar el programa de continuidad acad&eacute;mica remota para salvaguardar la salud y la vida ya que aislar a las personas parece ser la medida &oacute;ptima para prevenir la propagaci&oacute;n del virus, esto en ausencia de una cura definitiva en el corto plazo. Sin embargo, muchos estudiantes enfrentan grandes dificultades para acceder y permanecer en clase debido a limitaciones tecnol&oacute;gicas que afectan su educaci&oacute;n. El uso de plataformas de aprendizaje plantea varias preguntas, en particular:</p> <ul style="text-align: justify"> <li>&iquest;Qu&eacute; municipios tienen m&aacute;s dificultades para implementar el programa remoto de continuidad acad&eacute;mica?</li> <li>&iquest;Cu&aacute;l deber&iacute;a ser la priorizaci&oacute;n en los municipios para la inversi&oacute;n en programas de tecnolog&iacute;as de la informaci&oacute;n?</li> </ul> <p style="text-align: justify">Resolver estas preguntas ayuda a enfocarse en ciertas áreas geográficas para desarrollar programas de intervención específicos que logren mejorar la experiencia de aprendizaje y los resultados de los estudiantes de primaria y secundaria.</p>""",
    unsafe_allow_html=True)
    pass

if (selection == 'Estadisticas descriptivas'):
    st.sidebar.header('Periodo de estudio')

    subselection = st.sidebar.radio(
        "Acá usted puede seleccionar el periodo que desee analizar.",
        [
            '2014-2019',
            '2019 2º Semestre'
        ]
    )

    # st.markdown("""<p style="text-align: justify;"><strong><span style="color: #ff0000;">PENDIENTE LA EXPLICACI&Oacute;N AC&Aacute;</span></strong></p>""",unsafe_allow_html=True)
    st.markdown("""<p style="text-align: justify;">Ac&aacute; puedes encontrar algunas de las estadisticas de las base de datos. Haciendo uso del panel izquierdo puede seleccionar el periodo que deseas analizar.</p>""",unsafe_allow_html=True)

    if (subselection == '2014-2019'):
        components.iframe(
            "https://public.tableau.com/views/Estadisticas_Descrip_14_al_19/Mapas2?:showVizHome=no&:embed=true", scrolling=True, width=1000, height=900)
        pass

    if (subselection == '2019 2º Semestre'):
        components.iframe(
            "https://public.tableau.com/views/Dashboard_Icfes_v2/Departamento?:showVizHome=no&:embed=true", scrolling=True, width=1000, height=900)
        pass

    pass

if (selection == 'Modelo'):
    components.iframe(
        "https://public.tableau.com/views/Descrip_Modelo/Departamento?:showVizHome=no&:embed=true", scrolling=True, height=900)
    pass

if (selection == 'Mapa de la estimación'):
    st.markdown(
        """<p style="text-align: justify;">Ahora que ya se ha seleccionado un Umbral (el cual puede seguir modificando en el panel izquierda), puede ver los resultados goereferenciados. Puede filtrar los resultados con los controlos que encontrara en el panel izquierdo.</p>""",
        unsafe_allow_html= True
        )

    file = "ShapeMap/MGN_MPIO_POLITICO.shp"
    MapaDpto = geopandas.read_file(file, encoding='utf-8')
    MapaDpto['MPIO_CCDGO_C'] = pd.to_numeric(MapaDpto['DPTO_CCDGO'] + MapaDpto['MPIO_CCDGO'])

    MapaDpto = MapaDpto.join(Test.set_index('COLE_COD_MCPIO_UBICACION'), how = 'left', on = 'MPIO_CCDGO_C')
    MapaDpto.fillna(0, inplace = True)

    # st.write(Test.columns)
    Riesgo_all = sorted(MapaDpto.Riesgo_total.unique())
    RiesgoSelect = st.sidebar.selectbox(
        "Seleccione un nivel de Riesgo",
        ['Todos'] + Riesgo_all
    )
    if RiesgoSelect != 'Todos':
        MapaDpto = MapaDpto[MapaDpto.Riesgo_total == RiesgoSelect]
        pass

    DPTO_CNMBR_all = sorted(MapaDpto.DPTO_CNMBR.unique().astype(str))
    DPTO_CNMBR = st.sidebar.selectbox(
        "Seleccione un Departamento",
        ['Todos'] + DPTO_CNMBR_all
    )
    if DPTO_CNMBR != 'Todos':
        MapaDpto = MapaDpto[MapaDpto.DPTO_CNMBR == DPTO_CNMBR]
        # DataFilter = pd.DataFrame(MapaDpto.drop(columns='geometry'))
        # st.write(
        #     px.pie(
        #         DataFilter[DataFilter.Riesgo_total == RiesgoSelect],
        #         values='Intercepto',
        #         names='Riesgo_total'  # , title='Population of European continent'
        #     )
        # )
        pass

    VariableGraph = 'Riesgo_total'

    min_cn, max_cn = MapaDpto[VariableGraph].quantile([0.001,0.999]).apply(round, 2)

    colormap = branca.colormap.LinearColormap(
        colors=Colores,
        index= [0, 1, 2, 3],
        vmin = min_cn,
        vmax = max_cn
    )

    tile = "CartoDB positron"
    # st.sidebar.selectbox(
    #     label="Mapa base",
    #     options=["CartoDB positron", "CartoDB dark_matter", "OpenStreetMap", "Stamen Toner", "Stamen Terrain", "Stamen Watercolor"],
    #     index=0,
    # )

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
    pass

if (selection == 'Simulación de una intervención'):
    opciones = ['Análisis País', 'Análisis Departamento',
                'Análisis Municipio']  # una lista desplegable con los análisis
    analisis = st.sidebar.selectbox(
        'Por favor seleccione un análisis de la lista.', opciones)
    # Anàlisis por Pais
    if analisis == 'Análisis País':
        #-------------------------------------------------------------------------------
        # Entradas de usuario ConexMilHab
        #-------------------------------------------------------------------------------
        ColumnsIn = ['FAMI_TIENECOMPUTADOR', 'COLE_NATURALEZA', 'ConexMilHab']
        st.subheader("Análisis País")
        #-------------------------------------------------------------------------------
        # Se calcula los valores base con los datos del usuario
        #-------------------------------------------------------------------------------
        Vector_Base_Pais = Test[(Test['Ano'] == max(Test['Ano']))]
        Risk_Base_Pais = Vector_Base_Pais['Riesgo_total']

        InD_FAMI_TIENECOMPUTADOR = st.slider(
            label="Porcentaje Incremeno Familias con Computador",
            min_value=0,
            max_value=100,
            value=0,
            step=1
        )
        InD_COLE_NATURALEZA = st.slider(
            label="Porcentaje Incremeno Colegios Privados",
            min_value=0,
            max_value=100,
            value=0,
            step=1
        )
        InD_ConexMilHab = st.slider(
            label="Conexiones Incremeno Internet por Mil Habitantes",
            min_value=0,
            max_value=100,
            value=0,
            step=1
        )
        #-------------------------------------------------------------------------------
        # Se calcula los vectores de predicción base con los datos del usuario
        #-------------------------------------------------------------------------------
        Vector_Usuario_Pais = Vector_Base_Pais.copy()
        Vector_Usuario_Pais['FAMI_TIENECOMPUTADOR'] = np.where(Vector_Usuario_Pais['FAMI_TIENECOMPUTADOR']*(
            1+InD_FAMI_TIENECOMPUTADOR/100) > 1, 1, Vector_Usuario_Pais['FAMI_TIENECOMPUTADOR']*(1+InD_FAMI_TIENECOMPUTADOR/100))
        Vector_Usuario_Pais['COLE_NATURALEZA'] = np.where(Vector_Usuario_Pais['COLE_NATURALEZA']*(
            1+InD_COLE_NATURALEZA/100) > 1, 1, Vector_Usuario_Pais['COLE_NATURALEZA']*(1+InD_COLE_NATURALEZA/100))
        Vector_Usuario_Pais['ConexMilHab'] = np.where(Vector_Usuario_Pais['ConexMilHab']*(
            1+InD_ConexMilHab/100) > 1000, 1000, Vector_Usuario_Pais['ConexMilHab']*(1+InD_ConexMilHab/100))
        #-------------------------------------------------------------------------------
        # Estimación
        #-------------------------------------------------------------------------------
        Estimate = Vector_Usuario_Pais
        Estimate = Estimate[~Estimate.isin([np.nan, np.inf, -np.inf]).any(1)]

        Estimate['Intercepto'] = 1
        Estimate['riesgo_forest'] = rf_model.predict(Estimate[variables1])
        Estimate['riesgo_forest'] = np.where(
            Estimate.riesgo_forest < 0.5, 0, 1)
        Estimate['riesgo_regression'] = rf_modelD.predict(Estimate[variables1])
        Estimate['riesgo_logit'] = logit1_res.predict(Estimate[variables1])
        Estimate['riesgo_logit'] = np.where(Estimate.riesgo_logit < 0.5, 0, 1)
        Estimate['Riesgo_total'] = Estimate.riesgo_forest + \
            Estimate.riesgo_logit+Estimate.riesgo_regression

        #-------------------------------------------------------------------------------
        # Mapas
        #-------------------------------------------------------------------------------
        file = "ShapeMap/MGN_MPIO_POLITICO.shp"
        MapaDpto = geopandas.read_file(file)
        MapaDpto['MPIO_CCDGO_C'] = pd.to_numeric(
            MapaDpto['DPTO_CCDGO'] + MapaDpto['MPIO_CCDGO'])
        MapaDpto = MapaDpto.join(Estimate.set_index(
            'COLE_COD_MCPIO_UBICACION'), how='left', on='MPIO_CCDGO_C')
        MapaDpto.fillna(0, inplace=True)
        DataFilter = pd.DataFrame(MapaDpto.drop(columns='geometry'))
        VariableGraph = 'Riesgo_total'
        colormap = branca.colormap.LinearColormap(
            colors=Colores,
            #colors = ['#FFFFFF', '#6495ED', '#FFA500', '#FF4500'],['#9BCCEA', '#7BA0BD', '#587C95', '#37546B']
            index=[0, 1, 2, 3],
            vmin=0,
            vmax=3
        )

        tile = "CartoDB positron"
        # st.sidebar.selectbox(
        #     label="Mapa base",
        #     options=["CartoDB dark_matter", "OpenStreetMap", "Stamen Toner", "Stamen Terrain",
        #              "Stamen Watercolor", "CartoDB positron"],
        #     index=0,
        # )

        m_crime = folium.Map(
            location=[4.570868, -74.2973328],
            zoom_start=5,
            tiles=tile
        )

        def nombreestilo(x): return {
            'fillColor': colormap(x['properties'][VariableGraph]),
            'color': 'black',
            'weight': 0,
            'fillOpacity': 0.75
        }

        stategeo = folium.GeoJson(
            MapaDpto.to_json(),
            name='Riesgo SABER PRO - Colombia',
            style_function=nombreestilo,
            tooltip=folium.GeoJsonTooltip(
                fields=['DPTO_CNMBR', 'MPIO_CNMBR', VariableGraph],
                aliases=['Departamento', 'Municipio', 'Riesgo'],
                localize=True
            )
        ).add_to(m_crime)

        colormap.add_to(m_crime)

        folium_static(m_crime)
        st.write('La estimación del Departamento es:',
                 Estimate, 'Los datos del análisis.')
        pass
    # Anàlisis por Departamento
    if analisis == 'Análisis Departamento':
        #-------------------------------------------------------------------------------
        # Entradas de usuario ConexMilHab
        #-------------------------------------------------------------------------------
        Departamento_Sel = st.sidebar.selectbox(
            label="Filtro Departamento",
            options=Test['DEPARTAMENTO'].unique(),
            index=0,
        )
        ColumnsIn = ['FAMI_TIENECOMPUTADOR', 'COLE_NATURALEZA', 'ConexMilHab']
        st.subheader("Análisis Departamento")
        #-------------------------------------------------------------------------------
        # Se calcula los valores base con los datos del usuario
        #-------------------------------------------------------------------------------
        Vector_Base_Departamento = Test[(Test['DEPARTAMENTO'] == Departamento_Sel) & (
            Test['Ano'] == max(Test['Ano']))]
        Risk_Base_Departamento = Vector_Base_Departamento['Riesgo_total']

        InD_FAMI_TIENECOMPUTADOR = st.slider(
            label="Porcentaje Incremeno Familias con Computador",
            min_value=0,
            max_value=100,
            value=0,
            step=1
        )
        InD_COLE_NATURALEZA = st.slider(
            label="Porcentaje Incremeno Colegios Privados",
            min_value=0,
            max_value=100,
            value=0,
            step=1
        )
        InD_ConexMilHab = st.slider(
            label="Conexiones Incremeno Internet por Mil Habitantes",
            min_value=0,
            max_value=100,
            value=0,
            step=1
        )
        #-------------------------------------------------------------------------------
        # Se calcula los vectores de predicción base con los datos del usuario
        #-------------------------------------------------------------------------------
        Vector_Usuario_Departamento = Vector_Base_Departamento.copy()
        Vector_Usuario_Departamento['FAMI_TIENECOMPUTADOR'] = np.where(Vector_Usuario_Departamento['FAMI_TIENECOMPUTADOR']*(
            1+InD_FAMI_TIENECOMPUTADOR/100) > 1, 1, Vector_Usuario_Departamento['FAMI_TIENECOMPUTADOR']*(1+InD_FAMI_TIENECOMPUTADOR/100))
        Vector_Usuario_Departamento['COLE_NATURALEZA'] = np.where(Vector_Usuario_Departamento['COLE_NATURALEZA']*(
            1+InD_COLE_NATURALEZA/100) > 1, 1, Vector_Usuario_Departamento['COLE_NATURALEZA']*(1+InD_COLE_NATURALEZA/100))
        Vector_Usuario_Departamento['ConexMilHab'] = np.where(Vector_Usuario_Departamento['ConexMilHab']*(
            1+InD_ConexMilHab/100) > 1000, 1000, Vector_Usuario_Departamento['ConexMilHab']*(1+InD_ConexMilHab/100))
        #-------------------------------------------------------------------------------
        # Estimación
        #-------------------------------------------------------------------------------
        Estimate = Vector_Usuario_Departamento
        Estimate = Estimate[~Estimate.isin([np.nan, np.inf, -np.inf]).any(1)]

        Estimate['Intercepto'] = 1
        Estimate['riesgo_forest'] = rf_model.predict(Estimate[variables1])
        Estimate['riesgo_forest'] = np.where(
            Estimate.riesgo_forest < 0.5, 0, 1)
        Estimate['riesgo_regression'] = rf_modelD.predict(Estimate[variables1])
        Estimate['riesgo_logit'] = logit1_res.predict(Estimate[variables1])
        Estimate['riesgo_logit'] = np.where(Estimate.riesgo_logit < 0.5, 0, 1)
        Estimate['Riesgo_total'] = Estimate.riesgo_forest + \
            Estimate.riesgo_logit+Estimate.riesgo_regression

        #-------------------------------------------------------------------------------
        # Mapas
        #-------------------------------------------------------------------------------
        file = "ShapeMap/MGN_MPIO_POLITICO.shp"
        MapaDpto = geopandas.read_file(file)
        MapaDpto['MPIO_CCDGO_C'] = pd.to_numeric(
            MapaDpto['DPTO_CCDGO'] + MapaDpto['MPIO_CCDGO'])
        MapaDpto = MapaDpto.join(Estimate.set_index(
            'COLE_COD_MCPIO_UBICACION'), how='left', on='MPIO_CCDGO_C')
        MapaDpto.fillna(0, inplace=True)
        DataFilter = pd.DataFrame(MapaDpto.drop(columns='geometry'))
        VariableGraph = 'Riesgo_total'
        colormap = branca.colormap.LinearColormap(
            colors=Colores,
            #colors = ['#FFFFFF', '#6495ED', '#FFA500', '#FF4500'],['#9BCCEA', '#7BA0BD', '#587C95', '#37546B']
            index=[0, 1, 2, 3],
            vmin=0,
            vmax=3
        )

        tile = "CartoDB positron"
        # st.sidebar.selectbox(
        #     label="Mapa base",
        #     options=["CartoDB dark_matter", "OpenStreetMap", "Stamen Toner", "Stamen Terrain",
        #              "Stamen Watercolor", "CartoDB positron"],
        #     index=0,
        # )

        m_crime = folium.Map(
            location=[4.570868, -74.2973328],
            zoom_start=5,
            tiles=tile
        )

        def nombreestilo(x): return {
            'fillColor': colormap(x['properties'][VariableGraph]),
            'color': 'black',
            'weight': 0,
            'fillOpacity': 0.75
        }

        stategeo = folium.GeoJson(
            MapaDpto.to_json(),
            name='Riesgo SABER PRO - Colombia',
            style_function=nombreestilo,
            tooltip=folium.GeoJsonTooltip(
                fields=['DPTO_CNMBR', 'MPIO_CNMBR', VariableGraph],
                aliases=['Departamento', 'Municipio', 'Riesgo'],
                localize=True
            )
        ).add_to(m_crime)

        colormap.add_to(m_crime)

        folium_static(m_crime)
        st.write('La estimación del Departamento es:',
                Estimate, 'Los datos del análisis.')
        pass

    # Anàlisis por Municipio
    elif analisis == 'Análisis Municipio':
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
            options=Test[Test['DEPARTAMENTO'] ==
                        Departamento_Sel]['MUNICIPIO'].unique(),
            index=0,
        )
        ColumnsIn = ['FAMI_TIENECOMPUTADOR', 'COLE_NATURALEZA', 'ConexMilHab']

        st.subheader("Análisis Municipio")
        #-------------------------------------------------------------------------------
        # Se calcula los valores base con los datos del usuario
        #-------------------------------------------------------------------------------
        Vector_Base_Municipio = Test[(Test['MUNICIPIO'] == Municipio_Sel) & (
            Test['Ano'] == max(Test['Ano']))]
        Risk_Base_Municipio = Vector_Base_Municipio['Riesgo_total']

        In_FAMI_TIENECOMPUTADOR = st.slider(
            label="Porcentaje Familias con Computador",
            min_value=0,
            max_value=100,
            value=int(Vector_Base_Municipio['FAMI_TIENECOMPUTADOR']*100),
            step=1
        )
        In_COLE_NATURALEZA = st.slider(
            label="Porcentaje de Colegios Privados",
            min_value=0,
            max_value=100,
            value=int(Vector_Base_Municipio['COLE_NATURALEZA']*100),
            step=1
        )
        In_ConexMilHab = st.slider(
            label="Conexiones de Internet por Mil Habitantes",
            min_value=0,
            max_value=1000,
            value=int(Vector_Base_Municipio['ConexMilHab']),
            step=1
        )
        #-------------------------------------------------------------------------------
        # Se calcula los vectores de predicción base con los datos del usuario
        #-------------------------------------------------------------------------------
        Vector_Usuario_Municipio = Vector_Base_Municipio.copy()
        Vector_Usuario_Municipio[ColumnsIn] = [
            In_FAMI_TIENECOMPUTADOR/100, In_COLE_NATURALEZA/100, In_ConexMilHab]
        #-------------------------------------------------------------------------------
        # Estimación
        #-------------------------------------------------------------------------------
        Estimate = Vector_Usuario_Municipio
        Estimate = Estimate[~Estimate.isin([np.nan, np.inf, -np.inf]).any(1)]

        Estimate['Intercepto'] = 1
        Estimate['riesgo_forest'] = rf_model.predict(Estimate[variables1])
        Estimate['riesgo_forest'] = np.where(
            Estimate.riesgo_forest < 0.5, 0, 1)
        Estimate['riesgo_regression'] = rf_modelD.predict(Estimate[variables1])
        Estimate['riesgo_logit'] = logit1_res.predict(Estimate[variables1])
        Estimate['riesgo_logit'] = np.where(Estimate.riesgo_logit < 0.5, 0, 1)
        Estimate['Riesgo_total'] = Estimate.riesgo_forest + \
            Estimate.riesgo_logit+Estimate.riesgo_regression
        #-------------------------------------------------------------------------------
        # Mapas
        #-------------------------------------------------------------------------------
        file = "ShapeMap/MGN_MPIO_POLITICO.shp"
        MapaDpto = geopandas.read_file(file)
        MapaDpto['MPIO_CCDGO_C'] = pd.to_numeric(
            MapaDpto['DPTO_CCDGO'] + MapaDpto['MPIO_CCDGO'])
        MapaDpto = MapaDpto.join(Estimate.set_index(
            'COLE_COD_MCPIO_UBICACION'), how='left', on='MPIO_CCDGO_C')
        MapaDpto.fillna(0, inplace=True)
        DataFilter = pd.DataFrame(MapaDpto.drop(columns='geometry'))
        VariableGraph = 'Riesgo_total'
        colormap = branca.colormap.LinearColormap(
            colors=Colores,
            #colors = ['#FFFFFF', '#6495ED', '#FFA500', '#FF4500'],['#9BCCEA', '#7BA0BD', '#587C95', '#37546B']
            index=[0, 1, 2, 3],
            vmin=0,
            vmax=3
        )

        tile = "CartoDB positron"
        # st.sidebar.selectbox(
        #     label="Mapa base",
        #     options=["CartoDB dark_matter", "OpenStreetMap", "Stamen Toner", "Stamen Terrain",
        #              "Stamen Watercolor", "CartoDB positron"],
        #     index=0,
        # )

        m_crime = folium.Map(
            location=[4.570868, -74.2973328],
            zoom_start=5,
            tiles=tile
        )

        def nombreestilo(x): return {
            'fillColor': colormap(x['properties'][VariableGraph]),
            'color': 'black',
            'weight': 0,
            'fillOpacity': 0.75
        }

        stategeo = folium.GeoJson(
            MapaDpto.to_json(),
            name='Riesgo SABER PRO - Colombia',
            style_function=nombreestilo,
            tooltip=folium.GeoJsonTooltip(
                fields=['DPTO_CNMBR', 'MPIO_CNMBR', VariableGraph],
                aliases=['Departamento', 'Municipio', 'Riesgo'],
                localize=True
            )
        ).add_to(m_crime)

        colormap.add_to(m_crime)

        folium_static(m_crime)
        st.write('La estimación del Municipio es:',
                Estimate, 'Los datos del análisis.')
        pass
    pass

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

    st.markdown(
        """<p style="text-align: justify;"><strong><span style="color: #ff0000;">PENDIENTE LA EXPLICACI&Oacute;N AC&Aacute;</span></strong></p>""",
        unsafe_allow_html=True)

    TypeGraph = st.sidebar.selectbox(
        label="Seleccione el tipo de gráfico que desea analizar",
        options=[
            "2D",
            '3D'
        ],
        index=0,
    )

    if (TypeGraph == "2D"):
        Column = st.sidebar.selectbox(
            label="Variable del eje y",
            options=[
                "ConexMilHab",
                'FAMI_TIENEINTERNET',
                'FAMI_TIENECOMPUTADOR',
                'ESTU_TIENEETNIA',
                'COLE_NATURALEZA',
                'PUNT_GLOBAL',
                'ContagioMilHab',
                "Indice_Rural"
                ],
            index=0,
        )
        Row = st.sidebar.selectbox(
            label="Variable del eje x",
            options=["Indice_Rural", "ConexMilHab", 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'ESTU_TIENEETNIA', 'COLE_NATURALEZA', 'PUNT_GLOBAL', 'ContagioMilHab'],
            index=0,
        )
        Size = st.sidebar.selectbox(
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
        pass
    else:
        Column = st.sidebar.selectbox(
            label="Variable del eje y",
            options=[
                "ConexMilHab",
                'FAMI_TIENEINTERNET',
                'FAMI_TIENECOMPUTADOR',
                'ESTU_TIENEETNIA',
                'COLE_NATURALEZA',
                'PUNT_GLOBAL',
                'ContagioMilHab',
                "Indice_Rural"
            ],
            index=0,
        )
        Row = st.sidebar.selectbox(
            label="Variable del eje x",
            options=["Indice_Rural", "ConexMilHab", 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR',
                     'ESTU_TIENEETNIA', 'COLE_NATURALEZA', 'PUNT_GLOBAL', 'ContagioMilHab'],
            index=0,
        )
        Fondo = st.sidebar.selectbox(
            label="Variable del eje x",
            options=['FAMI_TIENEINTERNET', "Indice_Rural", "ConexMilHab", 'FAMI_TIENECOMPUTADOR',
                     'ESTU_TIENEETNIA', 'COLE_NATURALEZA', 'PUNT_GLOBAL', 'ContagioMilHab'],
            index=0,
        )
        Size = st.sidebar.selectbox(
            label="Variable del tamaño",
            options=["ContagioMilHab", "ConexMilHab", 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR',
                     'ESTU_TIENEETNIA', 'COLE_NATURALEZA', 'PUNT_GLOBAL', "Indice_Rural"],
            index=0,
        )

        fig = px.scatter_3d(
            covid_19[covid_19.ContagioMilHab > 1],
            x=Row,
            y=Column,
            z=Fondo,
            hover_name="MUNICIPIO",
            color="Riesgo_total",
            color_continuous_scale=Colores,
            size=Size,
            # log_y=BoolY,
            # log_x=BoolX,
        )

        st.plotly_chart(fig.update_traces( mode='markers', marker_line_width=1.5))
        pass


    # st.write(covid_19.columns)
    pass

if selection == 'Conclusiones':
    st.markdown("HOLA")
    pass

if (selection == 'Modelo dinámico (Avanzado)'):
    subselection = st.sidebar.radio(
    "Ir a la subsección",
    [
        'Modelo',
        'Mapa de la estimación',
        'Simulación de una intervención',
        'Vulneravilidad COVID-19'
    ]
    )

    # Crear base de datos
    Data_Base = pd.read_csv(
        "https://raw.githubusercontent.com/IngFrustrado/AppDS4A/master/Data_Base_1419.csv",
        encoding='UTF-8'
        )

    UmbralDefault = math.ceil(Data_Base[Data_Base.Ano == 2019]['PUNT_GLOBAL'].mean() - 1 * Data_Base[Data_Base.Ano == 2019]['PUNT_GLOBAL'].std())
    
    risk = st.sidebar.slider(
        label="Puntaje Umbral",
        min_value=math.ceil(
            Data_Base[Data_Base.Ano == 2019]['PUNT_GLOBAL'].quantile(0.05)),
        max_value=math.ceil(UmbralDefault),
        value=math.ceil(Data_Base[Data_Base.Ano == 2019]
                        ['PUNT_GLOBAL'].quantile(0.5)),
        step=1
    )

    QuanUmbral = stats.percentileofscore(Data_Base[Data_Base.Ano == 2019]['PUNT_GLOBAL'], risk) / 100

    Data_Base['Riesgo'] = 2
    for i in [2014, 2015, 2016, 2017, 2018, 2019]:
        riskFor = Data_Base[Data_Base.Ano == i]['PUNT_GLOBAL'].quantile(QuanUmbral)
        Data_Base['Riesgo'] = np.where(i == Data_Base.Ano, np.where(Data_Base['PUNT_GLOBAL'] < riskFor, 1, 0), Data_Base.Riesgo)

    if risk != math.ceil(UmbralDefault):
        st.sidebar.warning(
            'Nuestro análisis se llevó a cabo con una puntaje umbral de ' +
        str(
            math.ceil(UmbralDefault)
        ))
        pass

    Data_Base2 = Data_Base.copy()
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
    Test['riesgo_forest'] = rf_model.predict(Test[variables1])
    Test['riesgo_forest'] = np.where(Test.riesgo_forest < 0.5, 0, 1)
    Test['riesgo_regression'] = rf_modelD.predict(Test[variables1])
    # Test['riesgo_regression'] = np.where(Test.riesgo_regression < 0.5, 0, 1)
    Test['riesgo_logit'] = logit1_res.predict(Test[variables1])
    Test['riesgo_logit'] = np.where(Test.riesgo_logit < 0.5, 0, 1)

    Test['Riesgo_total'] = Test.riesgo_forest+Test.riesgo_logit+Test.riesgo_regression

    if (subselection == 'Modelo'):
        Anno = 2019
        # st.selectbox(
        #     label="Año",
        #     options=[2019, 2018, 2017, 2016, 2015, 2014],
        #     index=0,
        # )

        st.header('Selección del Umbral')

        st.markdown(
            """<p style="text-align: justify;">A continuaci&oacute;n usted puede seleccionar en el panel izquierdo el puntaje umbral con el cual se determinara si un municipio entra en riesgo o no, recuerde que el Umbral que usted seleccione en esta etapa determinara todo el analisis subsiguiente. Para ayudarle determinar el Umbral hemos dise&ntilde;ado graficos y estadisticos que muestra informaci&oacute;n relevante.</p>""",
            unsafe_allow_html=True
        )

        figHist = px.histogram(
            Data_Base2[Data_Base2.Ano == Anno],
            x="PUNT_GLOBAL",
            color="Riesgo",
            color_discrete_sequence=[Colores[1], Colores[3]],
            marginal="rug",
            hover_data=['PUNT_GLOBAL', 'MUNICIPIO', 'DEPARTAMENTO'],
            nbins=150
        )
        st.plotly_chart(figHist)

        # st.write(Data_Base2.columns)

        st.header('Analisis de resultados')

        st.markdown(
            """<p style="text-align: justify;">Nuestros tres modelos han sido entrenados con informaci&oacute;n de los a&ntilde;os 2014 y 2018, a continuaci&oacute;n puede encontrar los resultados de las estimaciones para el a&ntilde;o 2019, como notara en muchas ocasiones existen municipios que nuestros modelos han determinado con nivel de riesgo&nbsp;muy alto (3) intermedio-alto (2) e intermedio-bajo (1) a&uacute;n cuando en el a&ntilde;o 2019 tuvieron resultados superiores al umbral seleccionado (No Riesgo). Consideramos que estos municipios tienen caracteristicas que los hacen muy vulnerables. A continuaci&oacute;n puede comparar los municipios seg&uacute;n su nivel de riesgo por la variable que usted desee observar.</p>""",
            unsafe_allow_html=True
        )

        varY = st.selectbox(
            label="Variable del eje y",
            options=[
                "ConexMilHab",
                'FAMI_TIENEINTERNET',
                'FAMI_TIENECOMPUTADOR',
                'ESTU_TIENEETNIA',
                'COLE_NATURALEZA',
                'PUNT_GLOBAL',
                'ContagioMilHab',
                "Indice_Rural"
            ])

        fig = px.box(
            Test,
            x="Riesgo_total",
            y=varY,
            color="Riesgo",
            color_discrete_sequence=[Colores[1], Colores[3]]
            # title='Connectivity vs Year', labels={
            # "Ano": "Year",
            # "ConexMilHab": "Connectivity"}
        )

        st.plotly_chart(fig)
        pass

    if (subselection == 'Mapa de la estimación'):
        st.markdown(
            """<p style="text-align: justify;">Ahora que ya se ha seleccionado un Umbral (el cual puede seguir modificando en el panel izquierda), puede ver los resultados goereferenciados. Puede filtrar los resultados con los controlos que encontrara en el panel izquierdo.</p>""",
            unsafe_allow_html= True
            )

        file = "ShapeMap/MGN_MPIO_POLITICO.shp"
        MapaDpto = geopandas.read_file(file, encoding='utf-8')
        MapaDpto['MPIO_CCDGO_C'] = pd.to_numeric(MapaDpto['DPTO_CCDGO'] + MapaDpto['MPIO_CCDGO'])

        MapaDpto = MapaDpto.join(Test.set_index('COLE_COD_MCPIO_UBICACION'), how = 'left', on = 'MPIO_CCDGO_C')
        MapaDpto.fillna(0, inplace = True)

        # st.write(Test.columns)
        Riesgo_all = sorted(MapaDpto.Riesgo_total.unique())
        RiesgoSelect = st.sidebar.selectbox(
            "Seleccione un nivel de Riesgo",
            ['Todos'] + Riesgo_all
        )
        if RiesgoSelect != 'Todos':
            MapaDpto = MapaDpto[MapaDpto.Riesgo_total == RiesgoSelect]
            pass

        DPTO_CNMBR_all = sorted(MapaDpto.DPTO_CNMBR.unique().astype(str))
        DPTO_CNMBR = st.sidebar.selectbox(
            "Seleccione un Departamento",
            ['Todos'] + DPTO_CNMBR_all
        )
        if DPTO_CNMBR != 'Todos':
            MapaDpto = MapaDpto[MapaDpto.DPTO_CNMBR == DPTO_CNMBR]
            # DataFilter = pd.DataFrame(MapaDpto.drop(columns='geometry'))
            # st.write(
            #     px.pie(
            #         DataFilter[DataFilter.Riesgo_total == RiesgoSelect],
            #         values='Intercepto',
            #         names='Riesgo_total'  # , title='Population of European continent'
            #     )
            # )
            pass

        VariableGraph = 'Riesgo_total'

        min_cn, max_cn = MapaDpto[VariableGraph].quantile([0.001,0.999]).apply(round, 2)

        colormap = branca.colormap.LinearColormap(
            colors=Colores,
            index= [0, 1, 2, 3],
            vmin = min_cn,
            vmax = max_cn
        )

        tile = "CartoDB positron"
        # st.sidebar.selectbox(
        #     label="Mapa base",
        #     options=["CartoDB positron", "CartoDB dark_matter", "OpenStreetMap", "Stamen Toner", "Stamen Terrain", "Stamen Watercolor"],
        #     index=0,
        # )

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
        pass

    if (subselection == 'Simulación de una intervención'):
        opciones = ['Análisis País', 'Análisis Departamento',
                    'Análisis Municipio']  # una lista desplegable con los análisis
        analisis = st.sidebar.selectbox(
            'Por favor seleccione un análisis de la lista.', opciones)
        # Anàlisis por Pais
        if analisis == 'Análisis País':
            #-------------------------------------------------------------------------------
            # Entradas de usuario ConexMilHab
            #-------------------------------------------------------------------------------
            ColumnsIn = ['FAMI_TIENECOMPUTADOR', 'COLE_NATURALEZA', 'ConexMilHab']
            st.subheader("Análisis País")
            #-------------------------------------------------------------------------------
            # Se calcula los valores base con los datos del usuario
            #-------------------------------------------------------------------------------
            Vector_Base_Pais = Test[(Test['Ano'] == max(Test['Ano']))]
            Risk_Base_Pais = Vector_Base_Pais['Riesgo_total']

            InD_FAMI_TIENECOMPUTADOR = st.slider(
                label="Porcentaje Incremeno Familias con Computador",
                min_value=0,
                max_value=100,
                value=0,
                step=1
            )
            InD_COLE_NATURALEZA = st.slider(
                label="Porcentaje Incremeno Colegios Privados",
                min_value=0,
                max_value=100,
                value=0,
                step=1
            )
            InD_ConexMilHab = st.slider(
                label="Conexiones Incremeno Internet por Mil Habitantes",
                min_value=0,
                max_value=100,
                value=0,
                step=1
            )
            #-------------------------------------------------------------------------------
            # Se calcula los vectores de predicción base con los datos del usuario
            #-------------------------------------------------------------------------------
            Vector_Usuario_Pais = Vector_Base_Pais.copy()
            Vector_Usuario_Pais['FAMI_TIENECOMPUTADOR'] = np.where(Vector_Usuario_Pais['FAMI_TIENECOMPUTADOR']*(
                1+InD_FAMI_TIENECOMPUTADOR/100) > 1, 1, Vector_Usuario_Pais['FAMI_TIENECOMPUTADOR']*(1+InD_FAMI_TIENECOMPUTADOR/100))
            Vector_Usuario_Pais['COLE_NATURALEZA'] = np.where(Vector_Usuario_Pais['COLE_NATURALEZA']*(
                1+InD_COLE_NATURALEZA/100) > 1, 1, Vector_Usuario_Pais['COLE_NATURALEZA']*(1+InD_COLE_NATURALEZA/100))
            Vector_Usuario_Pais['ConexMilHab'] = np.where(Vector_Usuario_Pais['ConexMilHab']*(
                1+InD_ConexMilHab/100) > 1000, 1000, Vector_Usuario_Pais['ConexMilHab']*(1+InD_ConexMilHab/100))
            #-------------------------------------------------------------------------------
            # Estimación
            #-------------------------------------------------------------------------------
            Estimate = Vector_Usuario_Pais
            Estimate = Estimate[~Estimate.isin([np.nan, np.inf, -np.inf]).any(1)]

            Estimate['Intercepto'] = 1
            Estimate['riesgo_forest'] = rf_model.predict(Estimate[variables1])
            Estimate['riesgo_forest'] = np.where(
                Estimate.riesgo_forest < 0.5, 0, 1)
            Estimate['riesgo_regression'] = rf_modelD.predict(Estimate[variables1])
            Estimate['riesgo_logit'] = logit1_res.predict(Estimate[variables1])
            Estimate['riesgo_logit'] = np.where(Estimate.riesgo_logit < 0.5, 0, 1)
            Estimate['Riesgo_total'] = Estimate.riesgo_forest + \
                Estimate.riesgo_logit+Estimate.riesgo_regression

            #-------------------------------------------------------------------------------
            # Mapas
            #-------------------------------------------------------------------------------
            file = "ShapeMap/MGN_MPIO_POLITICO.shp"
            MapaDpto = geopandas.read_file(file)
            MapaDpto['MPIO_CCDGO_C'] = pd.to_numeric(
                MapaDpto['DPTO_CCDGO'] + MapaDpto['MPIO_CCDGO'])
            MapaDpto = MapaDpto.join(Estimate.set_index(
                'COLE_COD_MCPIO_UBICACION'), how='left', on='MPIO_CCDGO_C')
            MapaDpto.fillna(0, inplace=True)
            DataFilter = pd.DataFrame(MapaDpto.drop(columns='geometry'))
            VariableGraph = 'Riesgo_total'
            colormap = branca.colormap.LinearColormap(
                colors=Colores,
                #colors = ['#FFFFFF', '#6495ED', '#FFA500', '#FF4500'],['#9BCCEA', '#7BA0BD', '#587C95', '#37546B']
                index=[0, 1, 2, 3],
                vmin=0,
                vmax=3
            )

            tile = "CartoDB positron"
            # st.sidebar.selectbox(
            #     label="Mapa base",
            #     options=["CartoDB dark_matter", "OpenStreetMap", "Stamen Toner", "Stamen Terrain",
            #              "Stamen Watercolor", "CartoDB positron"],
            #     index=0,
            # )

            m_crime = folium.Map(
                location=[4.570868, -74.2973328],
                zoom_start=5,
                tiles=tile
            )

            def nombreestilo(x): return {
                'fillColor': colormap(x['properties'][VariableGraph]),
                'color': 'black',
                'weight': 0,
                'fillOpacity': 0.75
            }

            stategeo = folium.GeoJson(
                MapaDpto.to_json(),
                name='Riesgo SABER PRO - Colombia',
                style_function=nombreestilo,
                tooltip=folium.GeoJsonTooltip(
                    fields=['DPTO_CNMBR', 'MPIO_CNMBR', VariableGraph],
                    aliases=['Departamento', 'Municipio', 'Riesgo'],
                    localize=True
                )
            ).add_to(m_crime)

            colormap.add_to(m_crime)

            folium_static(m_crime)
            st.write('La estimación del Departamento es:',
                    Estimate, 'Los datos del análisis.')
            pass
        # Anàlisis por Departamento
        if analisis == 'Análisis Departamento':
            #-------------------------------------------------------------------------------
            # Entradas de usuario ConexMilHab
            #-------------------------------------------------------------------------------
            Departamento_Sel = st.sidebar.selectbox(
                label="Filtro Departamento",
                options=Test['DEPARTAMENTO'].unique(),
                index=0,
            )
            ColumnsIn = ['FAMI_TIENECOMPUTADOR', 'COLE_NATURALEZA', 'ConexMilHab']
            st.subheader("Análisis Departamento")
            #-------------------------------------------------------------------------------
            # Se calcula los valores base con los datos del usuario
            #-------------------------------------------------------------------------------
            Vector_Base_Departamento = Test[(Test['DEPARTAMENTO'] == Departamento_Sel) & (
                Test['Ano'] == max(Test['Ano']))]
            Risk_Base_Departamento = Vector_Base_Departamento['Riesgo_total']

            InD_FAMI_TIENECOMPUTADOR = st.slider(
                label="Porcentaje Incremeno Familias con Computador",
                min_value=0,
                max_value=100,
                value=0,
                step=1
            )
            InD_COLE_NATURALEZA = st.slider(
                label="Porcentaje Incremeno Colegios Privados",
                min_value=0,
                max_value=100,
                value=0,
                step=1
            )
            InD_ConexMilHab = st.slider(
                label="Conexiones Incremeno Internet por Mil Habitantes",
                min_value=0,
                max_value=100,
                value=0,
                step=1
            )
            #-------------------------------------------------------------------------------
            # Se calcula los vectores de predicción base con los datos del usuario
            #-------------------------------------------------------------------------------
            Vector_Usuario_Departamento = Vector_Base_Departamento.copy()
            Vector_Usuario_Departamento['FAMI_TIENECOMPUTADOR'] = np.where(Vector_Usuario_Departamento['FAMI_TIENECOMPUTADOR']*(
                1+InD_FAMI_TIENECOMPUTADOR/100) > 1, 1, Vector_Usuario_Departamento['FAMI_TIENECOMPUTADOR']*(1+InD_FAMI_TIENECOMPUTADOR/100))
            Vector_Usuario_Departamento['COLE_NATURALEZA'] = np.where(Vector_Usuario_Departamento['COLE_NATURALEZA']*(
                1+InD_COLE_NATURALEZA/100) > 1, 1, Vector_Usuario_Departamento['COLE_NATURALEZA']*(1+InD_COLE_NATURALEZA/100))
            Vector_Usuario_Departamento['ConexMilHab'] = np.where(Vector_Usuario_Departamento['ConexMilHab']*(
                1+InD_ConexMilHab/100) > 1000, 1000, Vector_Usuario_Departamento['ConexMilHab']*(1+InD_ConexMilHab/100))
            #-------------------------------------------------------------------------------
            # Estimación
            #-------------------------------------------------------------------------------
            Estimate = Vector_Usuario_Departamento
            Estimate = Estimate[~Estimate.isin([np.nan, np.inf, -np.inf]).any(1)]

            Estimate['Intercepto'] = 1
            Estimate['riesgo_forest'] = rf_model.predict(Estimate[variables1])
            Estimate['riesgo_forest'] = np.where(
                Estimate.riesgo_forest < 0.5, 0, 1)
            Estimate['riesgo_regression'] = rf_modelD.predict(Estimate[variables1])
            Estimate['riesgo_logit'] = logit1_res.predict(Estimate[variables1])
            Estimate['riesgo_logit'] = np.where(Estimate.riesgo_logit < 0.5, 0, 1)
            Estimate['Riesgo_total'] = Estimate.riesgo_forest + \
                Estimate.riesgo_logit+Estimate.riesgo_regression

            #-------------------------------------------------------------------------------
            # Mapas
            #-------------------------------------------------------------------------------
            file = "ShapeMap/MGN_MPIO_POLITICO.shp"
            MapaDpto = geopandas.read_file(file)
            MapaDpto['MPIO_CCDGO_C'] = pd.to_numeric(
                MapaDpto['DPTO_CCDGO'] + MapaDpto['MPIO_CCDGO'])
            MapaDpto = MapaDpto.join(Estimate.set_index(
                'COLE_COD_MCPIO_UBICACION'), how='left', on='MPIO_CCDGO_C')
            MapaDpto.fillna(0, inplace=True)
            DataFilter = pd.DataFrame(MapaDpto.drop(columns='geometry'))
            VariableGraph = 'Riesgo_total'
            colormap = branca.colormap.LinearColormap(
                colors=Colores,
                #colors = ['#FFFFFF', '#6495ED', '#FFA500', '#FF4500'],['#9BCCEA', '#7BA0BD', '#587C95', '#37546B']
                index=[0, 1, 2, 3],
                vmin=0,
                vmax=3
            )

            tile = "CartoDB positron"
            # st.sidebar.selectbox(
            #     label="Mapa base",
            #     options=["CartoDB dark_matter", "OpenStreetMap", "Stamen Toner", "Stamen Terrain",
            #              "Stamen Watercolor", "CartoDB positron"],
            #     index=0,
            # )

            m_crime = folium.Map(
                location=[4.570868, -74.2973328],
                zoom_start=5,
                tiles=tile
            )

            def nombreestilo(x): return {
                'fillColor': colormap(x['properties'][VariableGraph]),
                'color': 'black',
                'weight': 0,
                'fillOpacity': 0.75
            }

            stategeo = folium.GeoJson(
                MapaDpto.to_json(),
                name='Riesgo SABER PRO - Colombia',
                style_function=nombreestilo,
                tooltip=folium.GeoJsonTooltip(
                    fields=['DPTO_CNMBR', 'MPIO_CNMBR', VariableGraph],
                    aliases=['Departamento', 'Municipio', 'Riesgo'],
                    localize=True
                )
            ).add_to(m_crime)

            colormap.add_to(m_crime)

            folium_static(m_crime)
            st.write('La estimación del Departamento es:',
                    Estimate, 'Los datos del análisis.')
            pass

        # Anàlisis por Municipio
        elif analisis == 'Análisis Municipio':
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
                options=Test[Test['DEPARTAMENTO'] ==
                            Departamento_Sel]['MUNICIPIO'].unique(),
                index=0,
            )
            ColumnsIn = ['FAMI_TIENECOMPUTADOR', 'COLE_NATURALEZA', 'ConexMilHab']

            st.subheader("Análisis Municipio")
            #-------------------------------------------------------------------------------
            # Se calcula los valores base con los datos del usuario
            #-------------------------------------------------------------------------------
            Vector_Base_Municipio = Test[(Test['MUNICIPIO'] == Municipio_Sel) & (
                Test['Ano'] == max(Test['Ano']))]
            Risk_Base_Municipio = Vector_Base_Municipio['Riesgo_total']

            In_FAMI_TIENECOMPUTADOR = st.slider(
                label="Porcentaje Familias con Computador",
                min_value=0,
                max_value=100,
                value=int(Vector_Base_Municipio['FAMI_TIENECOMPUTADOR']*100),
                step=1
            )
            In_COLE_NATURALEZA = st.slider(
                label="Porcentaje de Colegios Privados",
                min_value=0,
                max_value=100,
                value=int(Vector_Base_Municipio['COLE_NATURALEZA']*100),
                step=1
            )
            In_ConexMilHab = st.slider(
                label="Conexiones de Internet por Mil Habitantes",
                min_value=0,
                max_value=1000,
                value=int(Vector_Base_Municipio['ConexMilHab']),
                step=1
            )
            #-------------------------------------------------------------------------------
            # Se calcula los vectores de predicción base con los datos del usuario
            #-------------------------------------------------------------------------------
            Vector_Usuario_Municipio = Vector_Base_Municipio.copy()
            Vector_Usuario_Municipio[ColumnsIn] = [
                In_FAMI_TIENECOMPUTADOR/100, In_COLE_NATURALEZA/100, In_ConexMilHab]
            #-------------------------------------------------------------------------------
            # Estimación
            #-------------------------------------------------------------------------------
            Estimate = Vector_Usuario_Municipio
            Estimate = Estimate[~Estimate.isin([np.nan, np.inf, -np.inf]).any(1)]

            Estimate['Intercepto'] = 1
            Estimate['riesgo_forest'] = rf_model.predict(Estimate[variables1])
            Estimate['riesgo_forest'] = np.where(
                Estimate.riesgo_forest < 0.5, 0, 1)
            Estimate['riesgo_regression'] = rf_modelD.predict(Estimate[variables1])
            Estimate['riesgo_logit'] = logit1_res.predict(Estimate[variables1])
            Estimate['riesgo_logit'] = np.where(Estimate.riesgo_logit < 0.5, 0, 1)
            Estimate['Riesgo_total'] = Estimate.riesgo_forest + \
                Estimate.riesgo_logit+Estimate.riesgo_regression
            #-------------------------------------------------------------------------------
            # Mapas
            #-------------------------------------------------------------------------------
            file = "ShapeMap/MGN_MPIO_POLITICO.shp"
            MapaDpto = geopandas.read_file(file)
            MapaDpto['MPIO_CCDGO_C'] = pd.to_numeric(
                MapaDpto['DPTO_CCDGO'] + MapaDpto['MPIO_CCDGO'])
            MapaDpto = MapaDpto.join(Estimate.set_index(
                'COLE_COD_MCPIO_UBICACION'), how='left', on='MPIO_CCDGO_C')
            MapaDpto.fillna(0, inplace=True)
            DataFilter = pd.DataFrame(MapaDpto.drop(columns='geometry'))
            VariableGraph = 'Riesgo_total'
            colormap = branca.colormap.LinearColormap(
                colors=Colores,
                #colors = ['#FFFFFF', '#6495ED', '#FFA500', '#FF4500'],['#9BCCEA', '#7BA0BD', '#587C95', '#37546B']
                index=[0, 1, 2, 3],
                vmin=0,
                vmax=3
            )

            tile = "CartoDB positron"
            # st.sidebar.selectbox(
            #     label="Mapa base",
            #     options=["CartoDB dark_matter", "OpenStreetMap", "Stamen Toner", "Stamen Terrain",
            #              "Stamen Watercolor", "CartoDB positron"],
            #     index=0,
            # )

            m_crime = folium.Map(
                location=[4.570868, -74.2973328],
                zoom_start=5,
                tiles=tile
            )

            def nombreestilo(x): return {
                'fillColor': colormap(x['properties'][VariableGraph]),
                'color': 'black',
                'weight': 0,
                'fillOpacity': 0.75
            }

            stategeo = folium.GeoJson(
                MapaDpto.to_json(),
                name='Riesgo SABER PRO - Colombia',
                style_function=nombreestilo,
                tooltip=folium.GeoJsonTooltip(
                    fields=['DPTO_CNMBR', 'MPIO_CNMBR', VariableGraph],
                    aliases=['Departamento', 'Municipio', 'Riesgo'],
                    localize=True
                )
            ).add_to(m_crime)

            colormap.add_to(m_crime)

            folium_static(m_crime)
            st.write('La estimación del Municipio es:',
                    Estimate, 'Los datos del análisis.')
            pass
        pass

    if subselection == 'Vulneravilidad COVID-19':
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

        st.markdown(
            """<p style="text-align: justify;"><strong><span style="color: #ff0000;">PENDIENTE LA EXPLICACI&Oacute;N AC&Aacute;</span></strong></p>""",
            unsafe_allow_html=True)

        TypeGraph = st.sidebar.selectbox(
            label="Seleccione el tipo de gráfico que desea analizar",
            options=[
                "2D",
                '3D'
            ],
            index=0,
        )

        if (TypeGraph == "2D"):
            Column = st.sidebar.selectbox(
                label="Variable del eje y",
                options=[
                    "ConexMilHab",
                    'FAMI_TIENEINTERNET',
                    'FAMI_TIENECOMPUTADOR',
                    'ESTU_TIENEETNIA',
                    'COLE_NATURALEZA',
                    'PUNT_GLOBAL',
                    'ContagioMilHab',
                    "Indice_Rural"
                    ],
                index=0,
            )
            Row = st.sidebar.selectbox(
                label="Variable del eje x",
                options=["Indice_Rural", "ConexMilHab", 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'ESTU_TIENEETNIA', 'COLE_NATURALEZA', 'PUNT_GLOBAL', 'ContagioMilHab'],
                index=0,
            )
            Size = st.sidebar.selectbox(
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
            pass
        else:
            Column = st.sidebar.selectbox(
                label="Variable del eje y",
                options=[
                    "ConexMilHab",
                    'FAMI_TIENEINTERNET',
                    'FAMI_TIENECOMPUTADOR',
                    'ESTU_TIENEETNIA',
                    'COLE_NATURALEZA',
                    'PUNT_GLOBAL',
                    'ContagioMilHab',
                    "Indice_Rural"
                ],
                index=0,
            )
            Row = st.sidebar.selectbox(
                label="Variable del eje x",
                options=["Indice_Rural", "ConexMilHab", 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR',
                        'ESTU_TIENEETNIA', 'COLE_NATURALEZA', 'PUNT_GLOBAL', 'ContagioMilHab'],
                index=0,
            )
            Fondo = st.sidebar.selectbox(
                label="Variable del eje x",
                options=['FAMI_TIENEINTERNET', "Indice_Rural", "ConexMilHab", 'FAMI_TIENECOMPUTADOR',
                        'ESTU_TIENEETNIA', 'COLE_NATURALEZA', 'PUNT_GLOBAL', 'ContagioMilHab'],
                index=0,
            )
            Size = st.sidebar.selectbox(
                label="Variable del tamaño",
                options=["ContagioMilHab", "ConexMilHab", 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR',
                        'ESTU_TIENEETNIA', 'COLE_NATURALEZA', 'PUNT_GLOBAL', "Indice_Rural"],
                index=0,
            )

            fig = px.scatter_3d(
                covid_19[covid_19.ContagioMilHab > 1],
                x=Row,
                y=Column,
                z=Fondo,
                hover_name="MUNICIPIO",
                color="Riesgo_total",
                color_continuous_scale=Colores,
                size=Size,
                # log_y=BoolY,
                # log_x=BoolX,
            )

            st.plotly_chart(fig.update_traces( mode='markers', marker_line_width=1.5))
            pass


        # st.write(covid_19.columns)
        pass
