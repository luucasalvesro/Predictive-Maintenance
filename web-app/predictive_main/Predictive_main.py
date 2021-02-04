import pickle
import math
import pandas as pd
import numpy as np

class Predictive_main(object):
    def __init__(self):
        self.scaler_x_reg = pickle.load(open('parametros/scaler_x_reg.pkl', 'rb'))
        self.scaler_y_reg = pickle.load(open('parametros/scaler_y_reg.pkl', 'rb'))

    def data_cleaning(self, df1):
        # Rename Colunas
        cols = ['asset_id', 'runtime']
        cols_setting = ['setting_{}'.format(i + 1) for i in range(3)]
        cols_tag = ['tag_{}'.format(i + 1) for i in range(21)]
        cols_final = cols + cols_setting + cols_tag

        df1.columns = cols_final

        # Drop colunas que não serão utilizadas
        cols_drop = ['setting_3', 'tag_1', 'tag_5', 'tag_10', 'tag_16', 'tag_18', 'tag_19']
        df1.drop(cols_drop, inplace=True, axis=1)

        return df1

    def data_preparation(self, df2):
        # Scaling
        cols_selec_reg = ['runtime', 'tag_2', 'tag_3', 'tag_4', 'tag_7', 'tag_8', 'tag_9', 'tag_11',
                          'tag_12', 'tag_13', 'tag_14', 'tag_15', 'tag_20', 'tag_21']

        cols_selec_cla = ['runtime', 'tag_2', 'tag_3', 'tag_4', 'tag_7', 'tag_8', 'tag_9', 'tag_11',
                          'tag_12', 'tag_13', 'tag_14', 'tag_15', 'tag_17', 'tag_20', 'tag_21']

        df2_reg = df2.copy()
        df2_cla = df2.copy()

        df2_reg[cols_selec_reg] = self.scaler_x_reg.fit_transform(df2_reg[cols_selec_reg])

        return df2_reg[cols_selec_reg], df2_cla[cols_selec_cla]

    def get_prediction(self, model_reg, model_cla, original_data, test_data_reg, test_data_cla):
        # Predição
        pred_reg = model_reg.predict(test_data_reg)
        pred_cla = model_cla.predict_proba(test_data_cla)[:,1]

        # Adicionando predições nos dados originais
        pred_reg = np.expm1(pred_reg)
        pred_reg = self.scaler_y_reg.inverse_transform(pred_reg.reshape(-1,1))

        original_data['prediction_reg'] = pred_reg
        original_data['prediction_cla'] = pred_cla

        return original_data.to_json(orient = 'records', date_format = 'iso')
