import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc 
import plotly.graph_objects as go
import plotly.express as px


from datetime import datetime as dt
import json
import numpy as np
import pandas as pd

#Recall app
from app import app




#############################
# Load map data
#############################


##############################
#Map Layout
##############################
map=html.Div([
 #Place the main graph component here:
    html.Iframe(src=app.get_asset_url('Tab.html') , width = 1000,  height = 1000),
    # html.Img(src=app.get_asset_url('my-image.jpg'))
], className="ds4a-body")
    