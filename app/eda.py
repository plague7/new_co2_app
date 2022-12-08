import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import app.Utils as Utils
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def computeMiScores(X, y):
    X = X.copy()
    for col in X.select_dtypes(["object", "category"]):
        X[col], _ = X[col].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(
        X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def init():
    st.title('Exploratory Data Analysis')

    df = Utils.load_data()
    df_corr = df.corr()

    layout = {
        "title": "Confusion Matrix",
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"}
    }

    fig = px.imshow(df_corr,
                    title='Confusion matrix')

    fig.update_layout(autosize=True, width=500, height=600,
                      showlegend=True)

    with st.expander('Confusion matrix'):
        st.plotly_chart(fig, use_container_width=True,
                        sharing='streamlit',
                        config={'displayModeBar': False})

    with st.expander('Feature importance scores :'):
        st.write('TO DO :)')
        # df = df.drop(columns={'Comments', 'Outlier'})
        # df = df.fillna(0)

        # X = df.copy()
        # y = X.pop('TotalGHGEmissions')
        # scores = computeMiScores(X, y)
        # st.write(scores)

    with st.expander('Model\'s weights :'):
        st.write('Only available if model is a linear regression')
        # df = Utils.load_data()
        # model = Utils.loadModel()

        # colors = ['Positive' if c > 0 else 'Negative' for c in model.coef_]
        # fig = px.bar(
        #     x=df.columns, y=model.coef_, color=colors,
        #     color_discrete_sequence=['red', 'blue'],
        #     labels=dict(x='Feature', y='Linear coefficient'),
        #     title='Weight of each feature for predicting petal width'
        # )
        # st.plotly_chart(fig, use_container_width=True,
        #                 sharing='streamlit',
        #                 config={'displayModeBar': False})
