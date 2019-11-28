from flask import Flask, render_template, url_for, request, redirect, send_file
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from pandas_datareader.data import YahooOptions
from flask import jsonify
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route('/lista', methods=['GET'])
def lista():
    start = datetime(2015, 1, 1)
    end = datetime(2015, 12, 1)
    facebook = web.DataReader('FB', 'yahoo', start, end)
    stocks = pd.DataFrame(facebook)
    data = []
    
    for info in range(0, len(stocks)):
        # data.append({'high': stocks['High'][info], 'low': stocks['Low'][info], 'open': stocks['Open'][info], 'close': stocks['Close'][info], 'volume': stocks['Volume'][info], 'adj_close': stocks['Adj Close'][info]})
        data.append({'high': stocks['High'][info], 'low': stocks['Low'][info], 'open': stocks['Open'][info], 'close': stocks['Close'][info], 'volume': stocks['Volume'][info]/100, 'adj_close': stocks['Adj Close'][info]})
    
    print(data)
    return jsonify(data)
    # return render_template('data.html', data=stocks)



    start = datetime(2015, 1, 1)
    end = datetime(2016, 1, 1)
    facebook = web.DataReader('FB', 'yahoo', start, end)
    df = pd.DataFrame(facebook)

    dfreg = df.loc[:,['Adj Close','Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    data = []

    for i in range(0, len(dfreg)):
        data.append({'adj_close': dfreg['Adj Close'][i], 'volume': dfreg['Volume'][i]/100, 'hl_pct': dfreg['HL_PCT'][i], 'pct_change': dfreg['PCT_change'][i]})
    print(dfreg.head())
    return jsonify(data)


@app.route('/stockoff', methods=['GET'])
def stockoff():
    # content = request.json

    # date_ini = content['date_ini'].split('-')
    # date_fin = content['date_fin'].split('-')

    # start = datetime(int(date_ini[2]), int(date_ini[1]), int(date_ini[0]))
    # end = datetime(int(date_fin[2]), int(date_fin[1]), int(date_fin[0]))

    # stock = web.DataReader(content['symbol'], 'yahoo', start, end)
    stocks = pd.read_csv('ford.csv')
    
    data = []
    
    for info in range(0, len(stocks)):
        # data.append({'high': stocks['High'][info], 'low': stocks['Low'][info], 'open': stocks['Open'][info], 'close': stocks['Close'][info], 'volume': stocks['Volume'][info], 'adj_close': stocks['Adj Close'][info]})
        data.append({'high': stocks['High'][info], 'low': stocks['Low'][info], 'open': stocks['Open'][info], 'close': stocks['Close'][info], 'volume': stocks['Volume'][info]/100, 'adj_close': stocks['Adj Close'][info]})
    
    print(data)
    return jsonify(data)
    # return render_template('data.html', data=stocks)



    start = datetime(2015, 1, 1)
    end = datetime(2016, 1, 1)
    facebook = web.DataReader('FB', 'yahoo', start, end)
    df = pd.DataFrame(facebook)

    dfreg = df.loc[:,['Adj Close','Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    data = []

    for i in range(0, len(dfreg)):
        data.append({'adj_close': dfreg['Adj Close'][i], 'volume': dfreg['Volume'][i]/100, 'hl_pct': dfreg['HL_PCT'][i], 'pct_change': dfreg['PCT_change'][i]})
    print(dfreg.head())
    return jsonify(data)


@app.route('/evaluateoff/<symbol>', methods=['GET'])
def evaluateoff(symbol):

    df = pd.read_csv('ford.csv')
    
    # Criando novo dataframe
    dfreg = df.loc[:,['Adj Close','Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    # Substituindo valore nulos
    dfreg.fillna(value=-99999, inplace=True)
    
    # Separando 1% dos dados para realizar a predição
    forecast_out = int(math.ceil(0.01 * len(dfreg)))
    
    # Separando o que irá ser predito
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))
    
    # Distribuindo os dados para a regressão linear
    X = preprocessing.scale(X)
    
    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    
    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Linear regression
    clfreg = LinearRegression(n_jobs=-1)
    clfreg.fit(X_train, y_train)
    
    # Quadratic Regression 2
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2.fit(X_train, y_train)

    # Quadratic Regression 3
    clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    clfpoly3.fit(X_train, y_train)

    # KNN Regression
    clfknn = KNeighborsRegressor(n_neighbors=2)
    clfknn.fit(X_train, y_train)

    confidencereg = clfreg.score(X_test, y_test)
    confidencepoly2 = clfpoly2.score(X_test,y_test)
    confidencepoly3 = clfpoly3.score(X_test,y_test)
    confidenceknn = clfknn.score(X_test, y_test)

    print(confidencereg)
    print(confidencepoly2)
    print(confidencepoly3)
    print(confidenceknn)

    # confidence = {'linear_regression': confidencereg, 'quadratic_regression_2': confidencepoly2, 'quadratic_regression_3': confidencepoly3, 'knn_regression': confidenceknn}

    forecast_set = clfreg.predict(X_lately)
    dfreg['Forecast'] = np.nan


    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + timedelta(days=1)

    for i in forecast_set:
        next_date = next_unix
        next_unix += timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
    
    dfreg['Adj Close'].tail(500).plot()
    dfreg['Forecast'].tail(500).plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.savefig('./img/graph.png', facebolor='w', edgecolor='w', orientation='portrait', format='png', transparent=False)

    # dados = {'grafico': './img/graph.png', 'resultados': confidence}
    dados = {'grafico': './img/graph.png', 'linear_regression': confidencereg, 'quadratic_regression_2': confidencepoly2, 'quadratic_regression_3': confidencepoly3, 'knn_regression': confidenceknn}
    
    return jsonify(dados)


@app.route('/teste_evaluate/<stock>/<ini_date>/<fin_date>', methods=['GET'])
def teste_evaluate(stock, ini_date, fin_date):

    date_ini = ini_date.split('-')
    date_fin = fin_date.split('-')
    # print('=+'*30)
    # print(date_ini)
    # print('=+'*30)

    start = datetime(int(date_ini[2]), int(date_ini[1]), int(date_ini[0]))
    end = datetime(int(date_fin[2]), int(date_fin[1]), int(date_fin[0]))

    # start = datetime(2016, 1, 1)
    # end = datetime(2017, 1, 1)

    stock = web.DataReader(stock, 'yahoo', start, end)
    df = pd.DataFrame(stock)
    
    # Criando novo dataframe
    dfreg = df.loc[:,['Adj Close','Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    # Substituindo valore nulos
    dfreg.fillna(value=-99999, inplace=True)
    
    # Separando 1% dos dados para realizar a predição
    forecast_out = int(math.ceil(0.01 * len(dfreg)))
    
    # Separando o que irá ser predito
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))
    
    # Distribuindo os dados para a regressão linear
    X = preprocessing.scale(X)
    
    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    
    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Linear regression
    clfreg = LinearRegression(n_jobs=-1)
    clfreg.fit(X_train, y_train)
    
    # Quadratic Regression 2
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2.fit(X_train, y_train)

    # Quadratic Regression 3
    clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    clfpoly3.fit(X_train, y_train)

    # KNN Regression
    clfknn = KNeighborsRegressor(n_neighbors=2)
    clfknn.fit(X_train, y_train)

    confidencereg = round(clfreg.score(X_test, y_test), 3)
    confidencepoly2 = round(clfpoly2.score(X_test,y_test), 3)
    confidencepoly3 = round(clfpoly3.score(X_test,y_test), 3)
    confidenceknn = round(clfknn.score(X_test, y_test), 3)

    # print(confidencereg)
    # print(confidencepoly2)
    # print(confidencepoly3)
    # print(confidenceknn)

    # confidence = {'linear_regression': confidencereg, 'quadratic_regression_2': confidencepoly2, 'quadratic_regression_3': confidencepoly3, 'knn_regression': confidenceknn}

    forecast_set = clfreg.predict(X_lately)
    dfreg['Forecast'] = np.nan


    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + timedelta(days=1)

    for i in forecast_set:
        next_date = next_unix
        next_unix += timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
    
    dfreg['Adj Close'].tail(500).plot()
    dfreg['Forecast'].tail(500).plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.savefig('./img/graph.png', facebolor='w', edgecolor='w', orientation='portrait', format='png', transparent=False)

    # dados = {'grafico': './img/graph.png', 'resultados': confidence}
    dados = {'grafico': './img/graph.png', 'linear_regression': confidencereg, 'quadratic_regression_2': confidencepoly2, 'quadratic_regression_3': confidencepoly3, 'knn_regression': confidenceknn}
    
    return jsonify(dados)

@app.route('/image')
def image():
    return send_file('img/graph.png', mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)