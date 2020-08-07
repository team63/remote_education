#-------------------------------------------------------------------------------
# Import the libraries
#-------------------------------------------------------------------------------
from datetime import datetime as dt
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

# Fechas Español
import locale
locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
from datetime import datetime as dt

# pRute = "/home/centos/AppDS4A-master/"
pRute  = ''

#-------------------------------------------------------------------------------
# Colours
#-------------------------------------------------------------------------------
# Colores = ['#FF4500', '#D75043', '#AF5A86', '#8765C9'] # Azules
# Colores = ['#FF4500', '#D75043', '#AF5A86', '#8765C9']  # Spider-Man
# Colores = ['#8765C9', '#AF7A86', '#D79043', '#FFA500', ]  # Amarillo-Morado
# Colores = ['#FF4500', '#E65F55', '#CC7AAA', '#B394FF']
# Colores = ['#B394FF', '#CC7AAA', '#E65F55', '#FF4500']
Colores = ['#FFD400', '#EB8F02', '#D84B04','#C40606' ]

#-------------------------------------------------------------------------------
# Dictionaries
#-------------------------------------------------------------------------------

NameVARH = {
    '% de familias que viven en un área rural': 'Indice_Rural',
    'Conexiones x1000 habitantes': 'ConexMilHab',
    '% de familias con internet': 'FAMI_TIENEINTERNET',
    '% de familias con computador': 'FAMI_TIENECOMPUTADOR',
    '% de estudiantes que pertenecen a una etnia': 'ESTU_TIENEETNIA',
    '% de Colegios Privados': 'COLE_NATURALEZA',
    'Promedio del puntaje ICFES': 'PUNT_GLOBAL',
    'No. de contagiados por COVID-19 x1000 habitantes': 'ContagioMilHab',
}

VarNameH = {
    'Ano': 'Año',
    'COLE_NATURALEZA': '% de Colegios Privados',
    'ConexMilHab': 'Conexiones x1000 habitantes',
    'DEPARTAMENTO': 'Departamento',
    'DPTO_CNMBR': 'Departamento',
    'ESTU_TIENEETNIA': '% de estudiantes que pertenecen a una etnia',
    'FAMI_TIENECOMPUTADOR': '% de familias con computador',
    'FAMI_TIENEINTERNET': '% de familias con internet',
    'Indice_Rural': '% de familias que viven en un área rural',
    'Intercepto': 'Constante',
    'MPIO_CCDGO': 'Código del municipio',
    'MPIO_CCDGO_C': 'Código del municipio',
    'MPIO_CCNCT': 'Código del municipio',
    'MPIO_CNMBR': 'Municipio',
    'MPIO_CRSLC': 'Código del municipio',
    'MPIO_NANO': 'Código del municipio',
    'MPIO_NAREA': 'Código del municipio',
    'MUNICIPIO': 'Municipio',
    'NoAccesosFijos': 'No. de accesos a internet fijo',
    'PUNT_GLOBAL': 'Promedio del puntaje ICFES',
    'PoblacionTotal': 'Población total',
    'Riesgo': 'Riesgo',
    'Riesgo_total': 'Vulnerabilidad',
    'riesgo_forest': 'Riesgo estimado por random forest',
    'riesgo_logit': 'Riesgo estimado por logit',
    'riesgo_regression': 'Riesgo estimado por random forest regression',
    'ContagioMilHab': 'No. de contagiados por COVID-19 x1000 habitantes',
}

#-------------------------------------------------------------------------------
# Streamlit - Part 1 
#-------------------------------------------------------------------------------
st.title('Educación a distancia en los tiempos del COVID-19')

st.sidebar.image(
    Image.open(pRute + 'img/Logo.png'),
    # caption='Sunrise by the mountains',
    use_column_width=True,
    format='PNG'
)

st.sidebar.title("Panel de Navegación")
selection = st.sidebar.radio(
    "Ir a",
    [
        'Introducción',
        'Estadisticas descriptivas',
        'Modelo',
        'Vulneravilidad COVID-19',
        'Simulación de una intervención',
        'Conclusiones',
        'Modelo dinámico (Avanzado)'
    ]
    )

#-------------------------------------------------------------------------------
# Model estimation
#-------------------------------------------------------------------------------
if (selection != 'Introducción') & (selection != 'Estadisticas descriptivas') & (selection != 'Conclusiones') & (selection != 'Modelo') & (selection != 'Modelo dinámico (Avanzado)'):
    # Data Base ----
    Data_Base = pd.read_csv(
        pRute + "Data_Base_1419.csv",
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

    # Logit model ----
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

    # Regression tree model ----
    rf_modelD = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_modelD.fit(Data_Base1[variables1], Data_Base1['Riesgo'])
    pscore_forestd = rf_modelD.predict(Data_Base1[variables1])
    Data_Base1['pscore_forestd'] = pscore_forestd

    # Random forest model ----
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rf_model.fit(Data_Base1[variables1], Data_Base1['Riesgo'])
    pscore_forest = rf_model.predict(Data_Base1[variables1])
    Data_Base1['pscore_forest'] = pscore_forest

    # Estimation 2019 ----
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
    pass

#-------------------------------------------------------------------------------
# Streamlit part 2
#-------------------------------------------------------------------------------
if (selection == 'Introducción'):
    st.image(
        Image.open(pRute + 'img/Front.jpg'),
        # caption='Sunrise by the mountains',
        use_column_width=True
    )
    st.markdown("""<p style="text-align: justify;"><span style="font-size:10px;text-align: justify;color: #808080;">Foto tomada de: <span style="text-decoration: underline;"><a style="color: #808080; text-decoration: underline;" href="https://www.mineducacion.gov.co/portal/Preescolar-basica-y-media/">MinEducaci&oacute;n</a></span>.</span></p>""",
    unsafe_allow_html=True)

    st.markdown("""<p style="text-align: justify;">Con la aparici&oacute;n del virus SARS-CoV-2 se ha hecho necesario implementar medidas que corten con la propagaci&oacute;n de este virus, donde el lavado de manos, el uso de tapabocas y el distanciamiento social han sido las formas m&aacute;s efectivas que se han encontrado para evitarlo. Por ello y con el fin de proteger a los alumnos se hizo necesario <span>implementar la educaci&oacute;n a distancia. </span></p> <p style="text-align: justify;">Sin embargo, muchos estudiantes enfrentan grandes dificultades para acceder y permanecer en clase debido a las limitaciones tecnol&oacute;gicas que tienen; con lo cual su proceso formativo se ve gravemente afectado. <span>Esta situaci&oacute;n plantea dos</span> preguntas de inter&eacute;s social:</p> <ul style="text-align: justify;"> <li>&iquest;Qu&eacute; municipios tienen m&aacute;s dificultades para implementar el programa de educaci&oacute;n a distancia?</li> <li>&iquest;Cu&aacute;l deber&iacute;a ser la priorizaci&oacute;n en los municipios para la inversi&oacute;n en programas de tecnolog&iacute;as de la informaci&oacute;n?</li> </ul> <p style="text-align: justify;">La respuesta de estas preguntas ayudar&aacute; a enfocar las pol&iacute;ticas sociales en ciertas &aacute;reas geogr&aacute;ficas, al dise&ntilde;o e implementaci&oacute;n de intervenciones espec&iacute;ficas, logrando as&iacute; mejorar la experiencia de aprendizaje y los resultados de los estudiantes de primaria y secundaria.</p> <p style="text-align: justify;">Para guiarlo se han desarrollado 7 secciones, en los cuales usted encontrar&aacute; elementos interactivos que le permitir&aacute; enfocar su an&aacute;lisis a sus &aacute;reas de inter&eacute;s.</p>""",
    unsafe_allow_html=True)
    pass

if (selection == 'Estadisticas descriptivas'):

    st.markdown("""<p style="text-align: justify;">Ac&aacute; se encuentran algunos mapas interactivos con algunas de las variables m&aacute;s interesantes que descubrimos en el proceso de modelamiento. Estas variables son el promedio del puntaje global del examen ICFES, las conexiones por cada mil habitantes y porcentaje de estudiantes que viven en &aacute;reas rurales.</p> <p style="text-align: justify;">Puede seleccionar un grupo de municipios o solo uno de ellos directamente sobre el mapa.</p>""", unsafe_allow_html=True)

    components.iframe(
        "https://public.tableau.com/views/Estadisticas_Descrip_14_al_19/Mapas2?:loadOrderID=1&:display_count=y&:showTabs=y&:showVizHome=no&:embed=true", scrolling=True, width=1024, height=795)

    pass

if (selection == 'Modelo'):
    st.markdown("""<p style="text-align: justify;">Ahora bien, la estimaci&oacute;n de la vulnerabilidad de cada uno de los municipios se ha establecido por 3 modelos (<em>logit</em>, <em>random forest</em> y <em>random forest regression</em>), donde cada uno de ellos recoge caracter&iacute;sticas distintas de la informaci&oacute;n reportada por el ICFES, MinTic y DANE.</p> <p style="text-align: justify;">Usted puede refinar los resultados de nuestra estimaci&oacute;n seleccionando municipios de manera individual.</p>""", unsafe_allow_html=True)

    components.iframe(
        "https://public.tableau.com/views/Descrip_Modelo/Departamento?:showVizHome=no&:embed=true", scrolling=True, width=1008, height=827)
    pass

if (selection == 'Vulneravilidad COVID-19'):
    url = "https://www.datos.gov.co/api/views/gt2j-8ykr/rows.csv?accessType=DOWNLOAD"
    DataCOVID = pd.read_csv(url)

    FechaMAX = max(pd.to_datetime(DataCOVID['fecha reporte web']).dt.date)

    filtroNoEstado = ['Leve', 'Moderado', 'Grave']
    DataCOVID = DataCOVID[DataCOVID['Estado'].isin(filtroNoEstado)]
    COVIDAgrup = DataCOVID.groupby(DataCOVID['Código DIVIPOLA']).count()[
        'ID de caso'].reset_index()
    COVIDAgrup = COVIDAgrup.rename(columns={"ID de caso": "CasosActivos"})

    CovidTest = Test.merge(
        COVIDAgrup, how='left', left_on='COLE_COD_MCPIO_UBICACION', right_on='Código DIVIPOLA')
    CovidTest['CasosActivos'] = CovidTest['CasosActivos'].fillna(0)
    CovidTest = CovidTest.drop(columns=[
                               'Ano', 'PoblacionTotal', 'NoAccesosFijos', 'Indice_Rural', 'Código DIVIPOLA'])

    Poblacion_2020 = pd.read_excel(pRute + "2020-poblacion.xlsx")  # .dropna()
    covid_19 = CovidTest.merge(Poblacion_2020[['Municipio', 'total', 'Rural']],
                               how='left', left_on='COLE_COD_MCPIO_UBICACION', right_on='Municipio')
    covid_19['ContagioMilHab'] = 1000 * \
        covid_19['CasosActivos'] / covid_19['total']
    covid_19['Indice_Rural'] = covid_19['Rural']/covid_19['total'].round(2)

    # st.write(DataCOVID['Estado'].unique())

    st.markdown(
        """<p style="text-align: justify;">Ahora bien, ac&aacute; usted encontrara una informaci&oacute;n con corte al """ +
        FechaMAX.strftime("%d %B %Y") + """ de los casos positivos sintom&aacute;ticos por municipio confirmados por el Instituto Nacional de Salud de Colombia.</p> <p style="text-align: justify;">Ac&aacute; usted puede observar los municipios con mayor vulnerabilidad, mayor n&uacute;mero de contagios por cada mil habitantes y otras variables de inter&eacute;s. Esto guarda relevancia puesto que los estudiantes pertenecientes &nbsp;a un municipio con altas tasas de contagio y con poco acceso a un computador o internet dif&iacute;cilmente puede participar en los programas de educaci&oacute;n a distancia.</p>""",
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
        Row = st.sidebar.selectbox(
            label="Variable del eje x",
            options=[
                '% de familias que viven en un área rural',
                '% de familias con internet',
                '% de familias con computador',
                '% de estudiantes que pertenecen a una etnia',
                '% de Colegios Privados',
                'Promedio del puntaje ICFES',
                'No. de contagiados por COVID-19 x1000 habitantes',
                'Conexiones x1000 habitantes',
                ],
            index=0,
        )
        if NameVARH[Row] in ['ContagioMilHab', 'ConexMilHab', 'PUNT_GLOBAL']:
            BoolX = True
            st.sidebar.warning(
                'La variable del eje X se muestra en escala logarítmica.')
            pass
        else:
            BoolX = False
            pass


        Column = st.sidebar.selectbox(
            label="Variable del eje y",
            options=[
                'Conexiones x1000 habitantes',
                'Promedio del puntaje ICFES',
                'No. de contagiados por COVID-19 x1000 habitantes',
                '% de familias con internet',
                '% de familias con computador',
                '% de estudiantes que pertenecen a una etnia',
                '% de Colegios Privados',
                '% de familias que viven en un área rural',
                ],
            index=0,
        )

        if NameVARH[Column] in ['ContagioMilHab', 'ConexMilHab', 'PUNT_GLOBAL']:
            BoolY = True
            st.sidebar.warning(
                'La variable del eje Y se muestra en escala logarítmica.')
            pass
        else:
            BoolY = False
            pass

        Size = st.sidebar.selectbox(
            label="Variable del tamaño",
            options=[
                'No. de contagiados por COVID-19 x1000 habitantes',
                'Conexiones x1000 habitantes',
                'Promedio del puntaje ICFES',
                '% de familias con internet',
                '% de familias con computador',
                '% de estudiantes que pertenecen a una etnia',
                '% de Colegios Privados',
                '% de familias que viven en un área rural',
                ],
            index=0,
        )

        fig = px.scatter(
            covid_19[covid_19.ContagioMilHab > 1],
            x=NameVARH[Row],
            y=NameVARH[Column],
            hover_name="MUNICIPIO",
            color="Riesgo_total",
            color_continuous_scale=Colores,
            size=NameVARH[Size],
            log_y=BoolY,
            log_x=BoolX,
        )

        fig['layout']['yaxis']['title']['text'] = Column
        fig['layout']['xaxis']['title']['text'] = Row
        fig['layout']['coloraxis']['colorbar']['title']['text'] = VarNameH["Riesgo_total"]

        fig.update_layout(
            autosize=True,
            margin=dict(
                b=100
            ),
            height=630,
            width=700
        )

        st.plotly_chart(fig.update_traces(
            mode='markers', marker_line_width=1.5))
        pass
    else:
        Row = st.sidebar.selectbox(
            label="Variable del eje x",
            options=[
                '% de familias que viven en un área rural',
                '% de familias con internet',
                '% de familias con computador',
                '% de estudiantes que pertenecen a una etnia',
                '% de Colegios Privados',
                'No. de contagiados por COVID-19 x1000 habitantes',
                'Conexiones x1000 habitantes',
                'Promedio del puntaje ICFES',
                ],
            index=0,
        )

        Column = st.sidebar.selectbox(
            label="Variable del eje y",
            options=[
                'Conexiones x1000 habitantes',
                'Promedio del puntaje ICFES',
                'No. de contagiados por COVID-19 x1000 habitantes',
                '% de familias con internet',
                '% de familias con computador',
                '% de estudiantes que pertenecen a una etnia',
                '% de Colegios Privados',
                '% de familias que viven en un área rural',
            ],
            index=0,
        )

        Fondo = st.sidebar.selectbox(
            label="Variable del eje z",
            options=[
                '% de familias con internet',
                '% de familias con computador',
                '% de estudiantes que pertenecen a una etnia',
                '% de Colegios Privados',
                '% de familias que viven en un área rural',
                'No. de contagiados por COVID-19 x1000 habitantes',
                'Conexiones x1000 habitantes',
                'Promedio del puntaje ICFES',
                ],
            index=0,
        )
        Size = st.sidebar.selectbox(
            label="Variable del tamaño",
            options=[
                'No. de contagiados por COVID-19 x1000 habitantes',
                'Conexiones x1000 habitantes',
                'Promedio del puntaje ICFES',
                '% de familias con internet',
                '% de familias con computador',
                '% de estudiantes que pertenecen a una etnia',
                '% de Colegios Privados',
                '% de familias que viven en un área rural',
                ],
            index=0,
        )

        fig3D = px.scatter_3d(
            covid_19[covid_19.ContagioMilHab > 1],
            x=NameVARH[Row],
            y=NameVARH[Column],
            z=NameVARH[Fondo],
            hover_name="MUNICIPIO",
            color="Riesgo_total",
            color_continuous_scale=Colores,
            size=NameVARH[Size],
            # log_y=BoolY,
            # log_x=BoolX,
        )

        # st.write(fig3D['layout'])

        # fig3D['layout']['yaxis']['title']['text'] = Column
        # fig3D['layout']['xaxis']['title']['text'] = Row
        # fig3D['layout']['zaxis']['title']['text'] = Fondo
        fig3D['layout']['coloraxis']['colorbar']['title']['text'] = VarNameH["Riesgo_total"]

        fig3D.update_layout(
            autosize=True,
            margin=dict(
                b=100
            ),
            height=630,
            width=700
        )

        st.plotly_chart(fig3D.update_traces(mode='markers', marker_line_width=1.5))
        pass

    # st.write(covid_19.columns)
    pass

if (selection == 'Simulación de una intervención'):
    st.markdown(
        """<p style="text-align: justify;">Ahora, queremos invitarlo a que realice una simulaci&oacute;n de 4 de las variables m&aacute;s representativas de nuestro modelo. Esta intervenci&oacute;n la puede realizar de manera nacional o focalizada por departamento o municipio seg&uacute;n sea de su inter&eacute;s.</p> <p style="text-align: justify;">Puede filtrar los resultados con los controles que encontrará en el panel izquierdo.</p>""",
        unsafe_allow_html= True
        )

    file = pRute + "ShapeMap/MGN_MPIO_POLITICO.shp"
    MapaDpto = geopandas.read_file(file, encoding='utf-8')
    MapaDpto['MPIO_CCDGO_C'] = pd.to_numeric(MapaDpto['DPTO_CCDGO'] + MapaDpto['MPIO_CCDGO'])

    MapaDpto = MapaDpto.join(Test.set_index('COLE_COD_MCPIO_UBICACION'), how = 'left', on = 'MPIO_CCDGO_C')
    MapaDpto.fillna(0, inplace = True)

    DPTO_CNMBR_all = sorted(MapaDpto.DPTO_CNMBR.unique().astype(str))
    DPTO_CNMBR = st.sidebar.selectbox(
        "Seleccione un Departamento",
        ['Todos'] + DPTO_CNMBR_all
    )
    
    #-------------------------------------------------------------------------------
    # Entradas de usuario ConexMilHab
    #-------------------------------------------------------------------------------
    ColumnsIn = ['FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 'COLE_NATURALEZA', 'ConexMilHab']
    st.subheader("Variables de intervención")
    #-------------------------------------------------------------------------------
    # Se calcula los valores base con los datos del usuario
    #-------------------------------------------------------------------------------
    Vector_Base_Pais = MapaDpto[(MapaDpto['Ano'] == max(Test['Ano']))]
    Risk_Base_Pais = Vector_Base_Pais['Riesgo_total']
    #-------------------------------------------------------------------------------
    # Controles de usaurio
    #-------------------------------------------------------------------------------
    InD_FAMI_TIENECOMPUTADOR = st.slider(
        label="Incremento en % de familias con computador",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01
    )
    # InD_FAMI_TIENEINTERNET = st.slider(
    #     label="Incremento en % de familias con internet",
    #     min_value=0.0,
    #     max_value=1.0,
    #     value=0.0,
    #     step=0.01
    # )
    InD_COLE_NATURALEZA = st.slider(
        label="Incremento en % de colegios privados",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01
    )
    InD_ConexMilHab = st.slider(
        label="Incremento de las conexiones de Internet por cada mil habitantes",
        min_value=0,
        max_value=500,
        value=0,
        step=1
    )
    #-------------------------------------------------------------------------------
    # Nuevos vectores
    #-------------------------------------------------------------------------------
    Vector_Usuario_Pais = Vector_Base_Pais.copy()
    Vector_Usuario_Pais['FAMI_TIENECOMPUTADOR'] = np.where(Vector_Base_Pais['FAMI_TIENECOMPUTADOR'] + InD_FAMI_TIENECOMPUTADOR > 1, 1, Vector_Base_Pais['FAMI_TIENECOMPUTADOR'] + InD_FAMI_TIENECOMPUTADOR)
    # Vector_Usuario_Pais['FAMI_TIENEINTERNET'] = np.where(Vector_Base_Pais['FAMI_TIENEINTERNET'] + InD_FAMI_TIENEINTERNET > 1, 1, Vector_Base_Pais['FAMI_TIENEINTERNET'] + InD_FAMI_TIENEINTERNET)
    Vector_Usuario_Pais['COLE_NATURALEZA'] = np.where(Vector_Base_Pais['COLE_NATURALEZA'] + InD_COLE_NATURALEZA > 1, 1, Vector_Base_Pais['COLE_NATURALEZA'] + InD_COLE_NATURALEZA)
    Vector_Usuario_Pais['ConexMilHab'] = np.where(Vector_Base_Pais['ConexMilHab'] + InD_ConexMilHab > 1000, 1000, Vector_Base_Pais['ConexMilHab'] + InD_ConexMilHab)
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
    MapaDpto = Estimate
    
    MPIO_CNMBR = 'Todos'

    if DPTO_CNMBR != 'Todos':
        MapaDpto = MapaDpto[MapaDpto.DPTO_CNMBR == DPTO_CNMBR]

        MPIO_CNMBR_all = sorted(MapaDpto.MPIO_CNMBR.unique().astype(str))
        MPIO_CNMBR = st.sidebar.selectbox(
            "Seleccione un Municipio",
            ['Todos'] + MPIO_CNMBR_all
        )
        if MPIO_CNMBR != 'Todos':
            MapaDpto = MapaDpto[MapaDpto.MPIO_CNMBR == MPIO_CNMBR]
            pass
        pass

    if (DPTO_CNMBR == 'Todos') or (MPIO_CNMBR == 'Todos'):
        Riesgo_all = sorted(MapaDpto.Riesgo_total.unique())
        RiesgoSelect = st.sidebar.selectbox(
            "Seleccione un nivel de vulnerabilidad",
            ['Todos'] + Riesgo_all
        )
        if RiesgoSelect != 'Todos':
            MapaDpto = MapaDpto[MapaDpto.Riesgo_total == RiesgoSelect]
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
            fields=[
                'DPTO_CNMBR',
                'MPIO_CNMBR',
                VariableGraph,
                'FAMI_TIENECOMPUTADOR',
                'FAMI_TIENEINTERNET',
                'COLE_NATURALEZA',
                'ConexMilHab'
                ],
            aliases = [
                VarNameH['DPTO_CNMBR'],
                VarNameH['MPIO_CNMBR'],
                VarNameH[VariableGraph],
                VarNameH['FAMI_TIENECOMPUTADOR'],
                VarNameH['FAMI_TIENEINTERNET'],
                VarNameH['COLE_NATURALEZA'],
                VarNameH['ConexMilHab']
                ], 
            localize = True
        )
    ).add_to(m_crime)
    colormap.add_to(m_crime)
    folium_static(m_crime)
    pass

if selection == 'Conclusiones':
    st.markdown(
        """<p style="text-align: justify;">Con las medidas de aislamiento por el COVID-19, que proh&iacute;ben a los estudiantes asistir a las aulas, la mayor parte del aprendizaje se ha realizado en l&iacute;nea, pero incluso aquellos alumnos que pueden conectarse con &eacute;xito es probable que se queden atr&aacute;s si no poseen la autodiciplina y la motivaci&oacute;n necesarias para la educaci&oacute;n a distancia. Adem&aacute;s, las investigaciones han demostrado que los estudiantes m&aacute;s pobres tienen peores resultados en los cursos en l&iacute;nea que en los cursos presenciales.</p> <p style="text-align: justify;">Nuestro an&aacute;lisis encontr&oacute; que las variables m&aacute;s relevantes relacionadas con el aumento de la vulnerabilidad acad&eacute;mica en Colombia son la conectividad por cada 1000 habitantes (como medida de acceso confiable a Internet de banda ancha) y la pertenencia a un grupo &eacute;tnico (minoritario) (asociado con bajos ingresos). Ciertos estados, en particular Amazonas, Vaup&eacute;s Guainia y Choc&oacute;, necesitan una intervenci&oacute;n seria, dados los altos valores mostrados para los factores de vulnerabilidad medidos. La centralizaci&oacute;n geogr&aacute;fica (asociada al aumento de la urbanizaci&oacute;n, medida por el &Iacute;ndice Rural), es tambi&eacute;n un fuerte predictor de mejores puntajes para los estudiantes.</p> <p style="text-align: justify;">Encontramos algunos municipios centralizados con alta vulnerabilidad como Coyaima en Tolima y Altos del Rosario en Bol&iacute;var, rodeados de municipios de cero vulnerabilidad. En particular estos municipios tienen bajo acceso a Internet y a la inform&aacute;tica y un alto &iacute;ndice rural y porcentaje de pertenencia a una etnia; mostrando una caracter&iacute;stica interesante de nuestro pa&iacute;s.</p> <p style="text-align: justify;">La mejora de los resultados acad&eacute;micos y la reducci&oacute;n de la gran brecha en el rendimiento acad&eacute;mico requiere que las intervenciones se centren en mantener las interacciones de los estudiantes (entre ellos y con sus profesores). Para que este escenario se haga realidad, es necesario que se den una serie de factores: las escuelas deben contar con los recursos para implementar programas de educaci&oacute;n a distancia, los estudiantes deben tener acceso a computadoras y conexiones confiables a Internet, y los padres deben tener la capacidad, el tiempo, la energ&iacute;a y la paciencia para convertirse en instructores de la escuela en casa.</p> <p style="text-align: justify;">Un esfuerzo concertado entre el gobierno estatal y los padres parece ser la estrategia m&aacute;s efectiva. Los proyectos complementarios podr&iacute;an incluir:</p> <ul> <li style="text-align: justify;">Optimizar las soluciones accesibles a los dispositivos m&oacute;viles, dado que los estudiantes tienen cierto acceso a ellos.</li> <li style="text-align: justify;">Ofreciendo planes de datos limitados (acceso concedido s&oacute;lo a sitios acad&eacute;micos, para evitar el mal uso).<br />Pol&iacute;ticas centradas en las minor&iacute;as &eacute;tnicas rurales, las comunidades con mayor vulnerabilidad.</li> <li style="text-align: justify;">Prestando computadoras a las familias.</li> <li style="text-align: justify;">Proporcionar est&iacute;mulos financieros a los municipios m&aacute;s vulnerables, a condici&oacute;n de que mejoren los resultados acad&eacute;micos en comparaci&oacute;n con el a&ntilde;o anterior.</li> <li style="text-align: justify;">Apoyo a los padres para que puedan ser profesores temporales.</li> </ul> <p style="text-align: justify;">Los desaf&iacute;os est&aacute;n relacionados con proporcionar a los estudiantes en situaci&oacute;n de vulnerabilidad las herramientas necesarias para el &eacute;xito, no s&oacute;lo computadoras port&aacute;tiles y acceso fiable a Internet de banda ancha, sino tambi&eacute;n para motivar a los padres a que ayuden a sus hijos a tener &eacute;xito acad&eacute;micamente.</p>""",
        unsafe_allow_html=True
        )
    pass

if (selection == 'Modelo dinámico (Avanzado)'):
    subselection = st.sidebar.radio(
    "Ir a la subsección",
    [
        'Modelo',
        'Mapa de la estimación',
        'Vulneravilidad COVID-19',
        'Simulación de una intervención',
    ]
    )

    # Crear base de datos
    Data_Base = pd.read_csv(
        pRute + "Data_Base_1419.csv",
        encoding='UTF-8'
        )

    UmbralDefault = math.ceil(Data_Base[Data_Base.Ano == 2019]['PUNT_GLOBAL'].mean() - 1 * Data_Base[Data_Base.Ano == 2019]['PUNT_GLOBAL'].std())
    
    risk = st.sidebar.slider(
        label="Puntaje Umbral",
        min_value=math.ceil(
            Data_Base[Data_Base.Ano == 2019]['PUNT_GLOBAL'].quantile(0.05)),
        max_value=math.ceil(Data_Base[Data_Base.Ano == 2019]
                        ['PUNT_GLOBAL'].quantile(0.5)),
        value=math.ceil(UmbralDefault),
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
            labels={'x': VarNameH['PUNT_GLOBAL']},
            color="Riesgo",
            color_discrete_sequence=[Colores[1], Colores[3]],
            marginal="rug",
            hover_data=['PUNT_GLOBAL', 'MUNICIPIO', 'DEPARTAMENTO'],
            nbins=150
        )

        figHist['layout']['yaxis']['title']['text'] = "Cuenta"
        figHist['layout']['xaxis']['title']['text'] = "Promedio del puntaje ICFES"

        st.plotly_chart(figHist)

        # st.write(Data_Base2.columns)

        st.header('Analisis de resultados')

        st.markdown(
            """<p style="text-align: justify;">Los modelos calculados (logit, random forest y random forest regression) han sido entrenados con informaci&oacute;n del ICFES de los periodos del 2014 al 2018 para determinar si un municipio est&aacute; en riesgo. Usando estos modelos estimamos el riesgo de cada municipio con la informaci&oacute;n del a&ntilde;o 2019. Cada modelo nos arroja como resultado 1 o 0 si se estima que puede caer en riesgo o no respectivamente. Sumando los resultados de los modelos, obtenemos el puntaje de vulnerabilidad para cada uno de los municipios. Recuerde que el puntaje l&iacute;mite con el que se determina si un municipio esta en riesgo o no es el que usted ha escogido en el panel izquierdo.</p> <p style="text-align: justify;">Sumando los resultados de los modelos, se han determinado niveles de vulnerabilidad: muy alta (3), intermedio-alta (2), intermedio-baja (1) y baja (0). A continuaci&oacute;n cuenta con un gr&aacute;fico de caja, donde puede seleccionar la variable que desea comparar discriminado por vulnerabilidad y riesgo (recuerde que un municipio se considera en riesgo si puntaje promedio es inferior al umbral seleccionado).</p>""",
            unsafe_allow_html=True
        )

        varY = st.selectbox(
            label="Variable del eje y",
            options=[
                'Conexiones x1000 habitantes',
                'Promedio del puntaje ICFES',
                '% de familias con internet',
                '% de familias con computador',
                '% de estudiantes que pertenecen a una etnia',
                '% de familias que viven en un área rural',
                '% de Colegios Privados',
            ])

        fig = px.box(
            Test,
            x="Riesgo_total",
            y=NameVARH[varY],
            labels={'x': "Vulnerabilidad", 'y': varY},
            color="Riesgo",
            color_discrete_sequence=[Colores[1], Colores[3]]
            # title='Connectivity vs Year', labels={
            # "Ano": "Year",
            # "ConexMilHab": "Connectivity"}
        )

        fig['layout']['yaxis']['title']['text'] = varY
        fig['layout']['xaxis']['title']['text'] = "Vulnerabilidad"

        st.plotly_chart(fig)
        pass

    if (subselection == 'Mapa de la estimación'):
        st.markdown(
            """<p style="text-align: justify;">Ahora que ya se ha seleccionado un puntaje promedio limite para determinar los municipios en riesgo (el cual puede seguir modificando en el panel izquierdo), puede ver los resultados del nivel de vulnerabilidad georreferenciados, el rojo representa los municipios con nivel de vulnerabilidad muy alto (3),&nbsp;el naranja oscuro intermedio-alta (2), el naranja claro intermedio-baja (1) y el amarillo baja (0).</p> <p style="text-align: justify;">Recuerde que puede filtrar los resultados con los controles que encontrara en el panel izquierdo.</p>""",
            unsafe_allow_html= True
            )

        file = pRute + "ShapeMap/MGN_MPIO_POLITICO.shp"
        MapaDpto = geopandas.read_file(file, encoding='utf-8')
        MapaDpto['MPIO_CCDGO_C'] = pd.to_numeric(MapaDpto['DPTO_CCDGO'] + MapaDpto['MPIO_CCDGO'])

        MapaDpto = MapaDpto.join(Test.set_index('COLE_COD_MCPIO_UBICACION'), how = 'left', on = 'MPIO_CCDGO_C')
        MapaDpto.fillna(0, inplace = True)

        Riesgo_all = sorted(MapaDpto.Riesgo_total.unique())
        RiesgoSelect = st.sidebar.selectbox(
            "Seleccione un nivel de vulnerabilidad",
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

    if (subselection == 'Vulneravilidad COVID-19'):
        url = "https://www.datos.gov.co/api/views/gt2j-8ykr/rows.csv?accessType=DOWNLOAD"
        DataCOVID = pd.read_csv(url)

        FechaMAX = max(pd.to_datetime(DataCOVID['fecha reporte web']).dt.date)

        filtroNoEstado = ['Leve', 'Moderado', 'Grave']
        DataCOVID = DataCOVID[DataCOVID['Estado'].isin(filtroNoEstado)]
        COVIDAgrup = DataCOVID.groupby(DataCOVID['Código DIVIPOLA']).count()[
            'ID de caso'].reset_index()
        COVIDAgrup = COVIDAgrup.rename(columns={"ID de caso": "CasosActivos"})

        CovidTest = Test.merge(
            COVIDAgrup, how='left', left_on='COLE_COD_MCPIO_UBICACION', right_on='Código DIVIPOLA')
        CovidTest['CasosActivos'] = CovidTest['CasosActivos'].fillna(0)
        CovidTest = CovidTest.drop(columns=[
                                'Ano', 'PoblacionTotal', 'NoAccesosFijos', 'Indice_Rural', 'Código DIVIPOLA'])

        Poblacion_2020 = pd.read_excel(pRute + "2020-poblacion.xlsx")  # .dropna()
        covid_19 = CovidTest.merge(Poblacion_2020[['Municipio', 'total', 'Rural']],
                                how='left', left_on='COLE_COD_MCPIO_UBICACION', right_on='Municipio')
        covid_19['ContagioMilHab'] = 1000 * \
            covid_19['CasosActivos'] / covid_19['total']
        covid_19['Indice_Rural'] = covid_19['Rural']/covid_19['total'].round(2)

        # st.write(DataCOVID['Estado'].unique())

        st.markdown(
            """<p style="text-align: justify;">Con la informaci&oacute;n a corte """ +
            FechaMAX.strftime("%d %B %Y") + """ de los casos positivos sintom&aacute;ticos por municipio confirmados por el Instituto Nacional de Salud de Colombia.</p> <p style="text-align: justify;">Pude desarrollar un analisis un analisis similar al desarrollado anteriormente pero ahora con el nuevo umbral seleccionado.</p>""",
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
            Row = st.sidebar.selectbox(
                label="Variable del eje x",
                options=[
                    '% de familias que viven en un área rural',
                    '% de familias con internet',
                    '% de familias con computador',
                    '% de estudiantes que pertenecen a una etnia',
                    '% de Colegios Privados',
                    'Promedio del puntaje ICFES',
                    'No. de contagiados por COVID-19 x1000 habitantes',
                    'Conexiones x1000 habitantes',
                    ],
                index=0,
            )
            if NameVARH[Row] in ['ContagioMilHab', 'ConexMilHab', 'PUNT_GLOBAL']:
                BoolX = True
                st.sidebar.warning(
                    'La variable del eje X se muestra en escala logarítmica.')
                pass
            else:
                BoolX = False
                pass


            Column = st.sidebar.selectbox(
                label="Variable del eje y",
                options=[
                    'Conexiones x1000 habitantes',
                    'Promedio del puntaje ICFES',
                    'No. de contagiados por COVID-19 x1000 habitantes',
                    '% de familias con internet',
                    '% de familias con computador',
                    '% de estudiantes que pertenecen a una etnia',
                    '% de Colegios Privados',
                    '% de familias que viven en un área rural',
                    ],
                index=0,
            )

            if NameVARH[Column] in ['ContagioMilHab', 'ConexMilHab', 'PUNT_GLOBAL']:
                BoolY = True
                st.sidebar.warning(
                    'La variable del eje Y se muestra en escala logarítmica.')
                pass
            else:
                BoolY = False
                pass

            Size = st.sidebar.selectbox(
                label="Variable del tamaño",
                options=[
                    'No. de contagiados por COVID-19 x1000 habitantes',
                    'Conexiones x1000 habitantes',
                    'Promedio del puntaje ICFES',
                    '% de familias con internet',
                    '% de familias con computador',
                    '% de estudiantes que pertenecen a una etnia',
                    '% de Colegios Privados',
                    '% de familias que viven en un área rural',
                    ],
                index=0,
            )

            fig = px.scatter(
                covid_19[covid_19.ContagioMilHab > 1],
                x=NameVARH[Row],
                y=NameVARH[Column],
                hover_name="MUNICIPIO",
                color="Riesgo_total",
                color_continuous_scale=Colores,
                size=NameVARH[Size],
                log_y=BoolY,
                log_x=BoolX,
            )

            fig['layout']['yaxis']['title']['text'] = Column
            fig['layout']['xaxis']['title']['text'] = Row
            fig['layout']['coloraxis']['colorbar']['title']['text'] = VarNameH["Riesgo_total"]

            fig.update_layout(
                autosize=True,
                margin=dict(
                    b=100
                ),
                height=630,
                width=700
            )

            st.plotly_chart(fig.update_traces(
                mode='markers', marker_line_width=1.5))
            pass
        else:
            Row = st.sidebar.selectbox(
                label="Variable del eje x",
                options=[
                    '% de familias que viven en un área rural',
                    '% de familias con internet',
                    '% de familias con computador',
                    '% de estudiantes que pertenecen a una etnia',
                    '% de Colegios Privados',
                    'No. de contagiados por COVID-19 x1000 habitantes',
                    'Conexiones x1000 habitantes',
                    'Promedio del puntaje ICFES',
                    ],
                index=0,
            )

            Column = st.sidebar.selectbox(
                label="Variable del eje y",
                options=[
                    'Conexiones x1000 habitantes',
                    'Promedio del puntaje ICFES',
                    'No. de contagiados por COVID-19 x1000 habitantes',
                    '% de familias con internet',
                    '% de familias con computador',
                    '% de estudiantes que pertenecen a una etnia',
                    '% de Colegios Privados',
                    '% de familias que viven en un área rural',
                ],
                index=0,
            )

            Fondo = st.sidebar.selectbox(
                label="Variable del eje z",
                options=[
                    '% de familias con internet',
                    '% de familias con computador',
                    '% de estudiantes que pertenecen a una etnia',
                    '% de Colegios Privados',
                    '% de familias que viven en un área rural',
                    'No. de contagiados por COVID-19 x1000 habitantes',
                    'Conexiones x1000 habitantes',
                    'Promedio del puntaje ICFES',
                    ],
                index=0,
            )
            Size = st.sidebar.selectbox(
                label="Variable del tamaño",
                options=[
                    'No. de contagiados por COVID-19 x1000 habitantes',
                    'Conexiones x1000 habitantes',
                    'Promedio del puntaje ICFES',
                    '% de familias con internet',
                    '% de familias con computador',
                    '% de estudiantes que pertenecen a una etnia',
                    '% de Colegios Privados',
                    '% de familias que viven en un área rural',
                    ],
                index=0,
            )

            fig3D = px.scatter_3d(
                covid_19[covid_19.ContagioMilHab > 1],
                x=NameVARH[Row],
                y=NameVARH[Column],
                z=NameVARH[Fondo],
                hover_name="MUNICIPIO",
                color="Riesgo_total",
                color_continuous_scale=Colores,
                size=NameVARH[Size],
                # log_y=BoolY,
                # log_x=BoolX,
            )

            # st.write(fig3D['layout'])

            # fig3D['layout']['yaxis']['title']['text'] = Column
            # fig3D['layout']['xaxis']['title']['text'] = Row
            # fig3D['layout']['zaxis']['title']['text'] = Fondo
            fig3D['layout']['coloraxis']['colorbar']['title']['text'] = VarNameH["Riesgo_total"]

            fig3D.update_layout(
                autosize=True,
                margin=dict(
                    b=100
                ),
                height=630,
                width=700
            )

            st.plotly_chart(fig3D.update_traces(mode='markers', marker_line_width=1.5))
            pass

        # st.write(covid_19.columns)
        pass

    if (subselection == 'Simulación de una intervención'):
        st.markdown(
            """<p style="text-align: justify;">Ahora que ya se ha seleccionado un Umbral (el cual puede seguir modificando en el panel izquierda), puede ver los resultados goereferenciados. Puede filtrar los resultados con los controlos que encontrara en el panel izquierdo.</p>""",
            unsafe_allow_html=True
        )

        file = pRute + "ShapeMap/MGN_MPIO_POLITICO.shp"
        MapaDpto = geopandas.read_file(file, encoding='utf-8')
        MapaDpto['MPIO_CCDGO_C'] = pd.to_numeric(MapaDpto['DPTO_CCDGO'] + MapaDpto['MPIO_CCDGO'])

        MapaDpto = MapaDpto.join(Test.set_index('COLE_COD_MCPIO_UBICACION'), how = 'left', on = 'MPIO_CCDGO_C')
        MapaDpto.fillna(0, inplace = True)

        DPTO_CNMBR_all = sorted(MapaDpto.DPTO_CNMBR.unique().astype(str))
        DPTO_CNMBR = st.sidebar.selectbox(
            "Seleccione un Departamento",
            ['Todos'] + DPTO_CNMBR_all
        )
        
        #-------------------------------------------------------------------------------
        # Entradas de usuario ConexMilHab
        #-------------------------------------------------------------------------------
        ColumnsIn = ['FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 'COLE_NATURALEZA', 'ConexMilHab']
        st.subheader("Variables de intervención")
        #-------------------------------------------------------------------------------
        # Se calcula los valores base con los datos del usuario
        #-------------------------------------------------------------------------------
        Vector_Base_Pais = MapaDpto[(MapaDpto['Ano'] == max(Test['Ano']))]
        Risk_Base_Pais = Vector_Base_Pais['Riesgo_total']
        #-------------------------------------------------------------------------------
        # Controles de usaurio
        #-------------------------------------------------------------------------------
        InD_FAMI_TIENECOMPUTADOR = st.slider(
            label="Incremento en % de familias con computador",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01
        )
        # InD_FAMI_TIENEINTERNET = st.slider(
        #     label="Incremento en % de familias con internet",
        #     min_value=0.0,
        #     max_value=1.0,
        #     value=0.0,
        #     step=0.01
        # )
        InD_COLE_NATURALEZA = st.slider(
            label="Incremento en % de colegios privados",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01
        )
        InD_ConexMilHab = st.slider(
            label="Incremento de las conexiones de Internet por cada mil habitantes",
            min_value=0,
            max_value=500,
            value=0,
            step=1
        )
        #-------------------------------------------------------------------------------
        # Nuevos vectores
        #-------------------------------------------------------------------------------
        Vector_Usuario_Pais = Vector_Base_Pais.copy()
        Vector_Usuario_Pais['FAMI_TIENECOMPUTADOR'] = np.where(Vector_Base_Pais['FAMI_TIENECOMPUTADOR'] + InD_FAMI_TIENECOMPUTADOR > 1, 1, Vector_Base_Pais['FAMI_TIENECOMPUTADOR'] + InD_FAMI_TIENECOMPUTADOR)
        # Vector_Usuario_Pais['FAMI_TIENEINTERNET'] = np.where(Vector_Base_Pais['FAMI_TIENEINTERNET'] + InD_FAMI_TIENEINTERNET > 1, 1, Vector_Base_Pais['FAMI_TIENEINTERNET'] + InD_FAMI_TIENEINTERNET)
        Vector_Usuario_Pais['COLE_NATURALEZA'] = np.where(Vector_Base_Pais['COLE_NATURALEZA'] + InD_COLE_NATURALEZA > 1, 1, Vector_Base_Pais['COLE_NATURALEZA'] + InD_COLE_NATURALEZA)
        Vector_Usuario_Pais['ConexMilHab'] = np.where(Vector_Base_Pais['ConexMilHab'] + InD_ConexMilHab > 1000, 1000, Vector_Base_Pais['ConexMilHab'] + InD_ConexMilHab)
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
        MapaDpto = Estimate
        
        MPIO_CNMBR = 'Todos'

        if DPTO_CNMBR != 'Todos':
            MapaDpto = MapaDpto[MapaDpto.DPTO_CNMBR == DPTO_CNMBR]

            MPIO_CNMBR_all = sorted(MapaDpto.MPIO_CNMBR.unique().astype(str))
            MPIO_CNMBR = st.sidebar.selectbox(
                "Seleccione un Municipio",
                ['Todos'] + MPIO_CNMBR_all
            )
            if MPIO_CNMBR != 'Todos':
                MapaDpto = MapaDpto[MapaDpto.MPIO_CNMBR == MPIO_CNMBR]
                pass
            pass

        if (DPTO_CNMBR == 'Todos') or (MPIO_CNMBR == 'Todos'):
            Riesgo_all = sorted(MapaDpto.Riesgo_total.unique())
            RiesgoSelect = st.sidebar.selectbox(
                "Seleccione un nivel de vulnerabilidad",
                ['Todos'] + Riesgo_all
            )
            if RiesgoSelect != 'Todos':
                MapaDpto = MapaDpto[MapaDpto.Riesgo_total == RiesgoSelect]
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
                fields=[
                    'DPTO_CNMBR',
                    'MPIO_CNMBR',
                    VariableGraph,
                    'FAMI_TIENECOMPUTADOR',
                    'FAMI_TIENEINTERNET',
                    'COLE_NATURALEZA',
                    'ConexMilHab'
                    ],
                aliases = [
                    VarNameH['DPTO_CNMBR'],
                    VarNameH['MPIO_CNMBR'],
                    VarNameH[VariableGraph],
                    VarNameH['FAMI_TIENECOMPUTADOR'],
                    VarNameH['FAMI_TIENEINTERNET'],
                    VarNameH['COLE_NATURALEZA'],
                    VarNameH['ConexMilHab']
                    ], 
                localize = True
            )
        ).add_to(m_crime)
        colormap.add_to(m_crime)
        folium_static(m_crime)
        pass
