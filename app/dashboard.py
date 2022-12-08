import streamlit as st
import pandas as pd
import plotly.express as px
import app.Utils as Utils


def plot(df, building_type_btn):
    updated_df = df[df['BuildingType'] == building_type_btn]

    fig = px.density_mapbox(updated_df, lat='Latitude',
                            lon='Longitude',
                            z='TotalGHGEmissions',
                            hover_data=['TotalGHGEmissions'],
                            opacity=.8,
                            radius=30,
                            mapbox_style='carto-positron')

    fig.update_layout(autosize=False, width=1300, height=400,
                      showlegend=True, margin={"l": 0, "r": 0, "t": 0, "b": 0})

    st.plotly_chart(fig, use_container_width=True,
                    sharing='streamlit',
                    config={'displayModeBar': False})


def init():
    df = Utils.load_data()
    st.title('Dashboard')

    building_type_btn = st.selectbox(
        label='Building type',
        options=df.BuildingType.unique()
    )
    plot(df, building_type_btn)
