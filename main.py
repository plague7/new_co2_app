import pandas as pd
import plotly.express as px

import streamlit as st
from app.Multipage import MultiPage
from app import dashboard, eda, demo

app = MultiPage()
app.add_page('Dashboard', dashboard.init)
app.add_page('EDA', eda.init)
app.add_page('Pred demo', demo.init)


# The main app
app.run()
