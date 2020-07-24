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

df = pd.read_csv('Data/superstore.csv', parse_dates=['Order Date', 'Ship Date'])

##############################################################
# SCATTER PLOT
###############################################################

###############################################################
# LINE PLOT
###############################################################


#################################################################################
# Here the layout for the plots to use.
#################################################################################
stats=html.Div([ 
	#Place the different graph components here.
  
	],className="ds4a-body")



