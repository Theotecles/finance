import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pyodbc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
 
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
 
if __name__ == '__main__':
   app.run_server(debug=True)