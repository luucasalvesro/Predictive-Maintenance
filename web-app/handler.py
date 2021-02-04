import os
import pickle
import pandas as pd

from flask                           import Flask, request, Response
from predictive_main.Predictive_main import Predictive_main

# Carregando o modelo
model_reg = pickle.load(open('model/reg_model.pkl', 'rb'))
model_cla = pickle.load(open('model/cla_model.pkl', 'rb'))

# Iniciando API
app = Flask(__name__)

@app.route('/maintenance/predict', methods=['POST'])
def maintenance_predict():
    test_json = request.get_json()

    if test_json: # Se tiver dados
        if isinstance(test_json, dict): # Exemplo unico
            test_raw = pd.DataFrame(test_json, index=[0])

        else: # Exemplo multiplo
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys())

        # Instanciando classe Predictive_main
        pipeline = Predictive_main()

        # Limpeza dos dados
        df1 = pipeline.data_cleaning(test_raw)

        # Preparação dos dados
        df2_reg, df2_cla = pipeline.data_preparation(df1)

        # Predição
        df_response = pipeline.get_prediction(model_reg, model_cla, test_raw, df2_reg, df2_cla)

        return df_response

    else:
        return Response('{}', status=200, mimetype = 'application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
