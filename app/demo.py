import streamlit as st
import numpy as np
import app.Utils as Utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
from sklearn.preprocessing import MinMaxScaler


def preprocess(df):
    # Encoding
    ohe = OneHotEncoder(
        sparse=False, handle_unknown='ignore')
    tmp_df = pd.DataFrame(ohe.fit_transform(
        df[['BuildingType', 'PrimaryPropertyType', 'Neighborhood']]), columns=ohe.get_feature_names())
    tmp_df.index = df.index

    df = df.drop(
        columns=['BuildingType', 'PrimaryPropertyType', 'Neighborhood'], axis=1)

    encoded_df = pd.concat([df, tmp_df], axis=1)

    T = ['CouncilDistrictCode', 'NumberofFloors', 'PropertyGFATotal',
         'PropertyGFAParking', 'ENERGYSTARScore', 'SteamUse(kBtu)',
         'BuildingAge', 'x0_Campus', 'x0_NonResidential',
         'x0_Nonresidential COS', 'x0_SPS-District K-12',
         'x1_Distribution Center', 'x1_Hospital', 'x1_Hotel', 'x1_K-12 School',
         'x1_Laboratory', 'x1_Large Office', 'x1_Medical Office',
         'x1_Mixed Use Property', 'x1_Office', 'x1_Other',
         'x1_Refrigerated Warehouse', 'x1_Residence Hall', 'x1_Restaurant',
         'x1_Retail Store', 'x1_Self-Storage Facility',
         'x1_Senior Care Community', 'x1_Small- and Mid-Sized Office',
         'x1_Supermarket / Grocery Store', 'x1_University', 'x1_Warehouse',
         'x1_Worship Facility', 'x2_BALLARD', 'x2_CENTRAL', 'x2_DELRIDGE',
         'x2_DOWNTOWN', 'x2_EAST', 'x2_GREATER DUWAMISH', 'x2_LAKE UNION',
         'x2_MAGNOLIA / QUEEN ANNE', 'x2_NORTH', 'x2_NORTHEAST', 'x2_NORTHWEST',
         'x2_SOUTHEAST', 'x2_SOUTHWEST']

    for feature in T:
        if feature not in encoded_df.columns:
            encoded_df[feature] = 0.0

    encoded_df = encoded_df.reindex(columns=T)
    data = np.array(encoded_df.values, dtype='float64').reshape(-1, 1)

    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)

    return scaled_data


def predict(model, data):
    data = np.array(data, dtype='float64').reshape(1, -1)
    return model.predict(data)[0]


def getInput(feature_name):
    if feature_name == 'BuildingType':
        return st.selectbox(
            feature_name,
            ('NonResidential', 'Nonresidential COS', 'SPS-District K-12',
             'Campus'))

    elif feature_name == 'PrimaryPropertyType':
        return st.selectbox(
            feature_name,
            ('Hotel', 'Other', 'Mixed Use Property', 'K-12 School',
             'University', 'Small- and Mid-Sized Office',
             'Self-Storage Facility', 'Warehouse', 'Large Office',
             'Senior Care Community', 'Medical Office', 'Retail Store',
             'Hospital', 'Residence Hall', 'Distribution Center',
             'Worship Facility', 'Supermarket / Grocery Store', 'Laboratory',
             'Refrigerated Warehouse', 'Restaurant', 'Office')
        )

    elif feature_name == 'Neighborhood':
        return st.selectbox(
            feature_name,
            ('DOWNTOWN', 'SOUTHEAST', 'NORTHEAST', 'EAST', 'NORTH',
             'MAGNOLIA / QUEEN ANNE', 'LAKE UNION', 'GREATER DUWAMISH',
             'BALLARD', 'NORTHWEST', 'CENTRAL', 'SOUTHWEST', 'DELRIDGE')
        )

    elif feature_name in ['NumberofFloors', 'BuildingAge', 'CouncilDistrictCode']:
        return st.number_input(feature_name, min_value=0, step=1)

    elif feature_name in ['PropertyGFATotal', 'PropertyGFAParking', 'ENERGYSTARScore', 'SteamUse(kBtu)', 'TotalGHGEmissions']:
        return st.text_input(feature_name)


def formatPrediction(y_pred, n):
    unit = 'lb CO2e/MWh'
    return '[{}] Total emission : {} {}'.format(n, str(y_pred), unit)


def init():
    st.title('Demo')

    df = Utils.load_data(exclude=True)
    model = Utils.loadModel()
    features_list = df.columns

    uploaded_file = st.file_uploader('Choose a file', type='csv')

    with st.expander('Or fill in your data :'):
        for feature in features_list:
            feature_var = '_feature_{}'.format(feature)
            globals()[feature_var] = getInput(feature)

    launch_prediction_btn = st.button(label='Predict')

    if launch_prediction_btn:
        if uploaded_file:
            df = Utils.load_data(uploaded_file, exclude=True)

            for i in df.index:
                df_row = pd.DataFrame(df.loc[i]).T
                data = preprocess(df_row)
                y_pred = predict(model, data)
                y_pred_str = formatPrediction(y_pred, i)
                st.write(y_pred_str)

        else:
            features_values = [v for k, v in globals().items()
                               if k.startswith('_feature_')]

            if not any(v == '' for v in features_values):
                df_dtypes = dict(df.dtypes)

                for i, l in enumerate(features_list):
                    v_dtype = df_dtypes[l]
                    if v_dtype == 'float64':
                        v = features_values[i]
                        features_values[i] = np.float64(v)

                features_values_df = pd.DataFrame(features_values).T
                features_values_df.columns = df.columns
                data = preprocess(features_values_df)
                y_pred = predict(model, data)

                y_pred_str = formatPrediction(y_pred)
                st.write(y_pred_str)

            else:
                st.write('Please provide a value for all the features.')
