import os
import requests
import json
import pandas as pd
import math

from flask import Flask, request, Response

# Token
TOKEN = '1444810854:AAHJcAk8uALT89MUYgsPNX9RLhXakHJu-IE'

# Info do bot
#https://api.telegram.org/bot1444810854:AAHJcAk8uALT89MUYgsPNX9RLhXakHJu-IE/getMe

# Get updates
#https://api.telegram.org/bot1444810854:AAHJcAk8uALT89MUYgsPNX9RLhXakHJu-IE/getUpdates

# Webhook local
#https://api.telegram.org/bot1444810854:AAHJcAk8uALT89MUYgsPNX9RLhXakHJu-IE/setWebhook?url=https://luuca-c64c5166.localhost.run

# Webhook Heroku
#https://api.telegram.org/bot1444810854:AAHJcAk8uALT89MUYgsPNX9RLhXakHJu-IE/setWebhook?url=https://maintenance-telegram-bot.herokuapp.com

# Send message
#https://api.telegram.org/bot1444810854:AAHJcAk8uALT89MUYgsPNX9RLhXakHJu-IE/sendMessage?chat_id=1400179405&text=Oi Lucas, estou bem, e você?


def send_message(chat_id, text):
    url = 'https://api.telegram.org/bot{}/'.format(TOKEN)
    url = url + 'sendMessage?chat_id={}'.format(chat_id)

    r = requests.post(url, json={'text': text})
    print('Status Code {}'.format(r.status_code))

    return None

def load_dataset(asset):
    # Carregando dados
    df_teste = pd.read_table('PM_test.txt',sep='\s+', header = None)

    # Criando as colunas
    cols = ['asset_id', 'runtime']
    cols_setting = ['setting_{}'.format(i + 1) for i in range(3)]
    cols_tag = ['tag_{}'.format(i + 1) for i in range(21)]
    cols_final = cols + cols_setting + cols_tag

    df_teste.columns = cols_final

    # Selecionando asset para predição
    df_teste = df_teste[df_teste['asset_id']==asset]

    if not df_teste.empty:
        # Convertendo dataframe para JSON
        data = json.dumps(df_teste.to_dict(orient='records'))
    else:
        data = 'error'

    return data

def predict(data):
    # Chamada da API
    url = 'https://predictive-main.herokuapp.com/maintenance/predict'
    header = {'Content-type': 'application/json' }
    data = data

    r = requests.post(url, data=data, headers=header)
    print('Status Code {}'.format(r.status_code ))

    d1 = pd.DataFrame(r.json(), columns=r.json()[0].keys())

    return d1

def parse_message(message):
    chat_id = message['message']['chat']['id']
    asset = message['message']['text']

    asset = asset.replace('/', '')

    try:
        asset = int(asset)

    except ValueError:
        asset = 'error'

    return chat_id, asset

# Inicialização da API
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.get_json()

        chat_id, asset = parse_message(message)

        if asset != 'error':
            # loading data
            data = load_dataset(asset)

            if data != 'error':
                # prediction
                d1 = predict(data)

                # calculation
                d2 = d1.iloc[-1]

                # send message
                msg = 'Probabilidade de falha do Ativo número {} é de {}% com {} ciclos restantes de vida útil '.format(
                                d2['asset_id'],
                                round(d2['prediction_cla']*100,3),
                                math.floor(d2['prediction_reg']))

                send_message(chat_id, msg)
                return Response('Ok', status=200)

            else:
                send_message(chat_id, 'Ativo não disponível, digite um número de Ativo de 1 a 100')
                return Response('Ok', status=200)

        else:
            send_message(chat_id, 'Por gentileza, digite um número de Ativo de 1 a 100')
            return Response('Ok', status=200)

    else:
        return '<h1> Maintenance Telegram Bot </h1>'


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
