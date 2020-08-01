import streamlit.components.v1 as components
components.iframe(
    "https://public.tableau.com/views/Dashboard_Icfes_v2/Departamento?:showVizHome=no&:embed=true", scrolling=True, width = 1000, height=900)
    
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
