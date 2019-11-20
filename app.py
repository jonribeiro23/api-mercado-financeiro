from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
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

# Cria o banco de dados
# app.config['SQLAlCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'

# Inicializa o banco
db = SQLAlchemy(app)


# classe que modela o banco
class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Task %r>' % self.id # retorna o id da tarefa


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        task_content = request.form['content']
        new_task = Todo(content=task_content)

        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect('/')
        except:
            return 'erro ao salvar no banco'

    else:
        tasks = Todo.query.order_by(Todo.date_created).all()
        return render_template('index.html', tasks=tasks)


@app.route('/delete/<int:id>')
def delete(id):
    task_to_delete = Todo.query.get_or_404(id)

    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'Erro ao deletar tarefa'


@app.route('/update/<int:id>', methods=['POST', 'GET'])
def update(id):
    task = Todo.query.get_or_404(id)
    if request.method == 'POST':
        task.content = request.form['content']
        try:
            db.session.commit()
            return redirect('/')
        except:
            return 'Erro ao atualizar a tarefa'
    else:
        return render_template('update.html', task=task)


@app.route('/lista')
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


@app.route('/predict')
def predict():
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


@app.route('/adjclose')
def adjclose():
    lm = LinearRegression(n_jobs=-1)
    start = datetime(2015, 1, 1)
    end = datetime(2016, 1, 1)
    facebook = web.DataReader('AAPL', 'yahoo', start, end)
    df = pd.DataFrame(facebook)
    df['sma'] = df['Close'].rolling(window=1).mean()
    
    y = df['Close']
    x = df[['High', 'Low', 'Open', 'Volume', 'Adj Close', 'sma']]

    # Dividindo os dados em treino e teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

    #Treinando o modelo
    lm.fit(x_train, y_train)

    #Fazendo a predição
    prediction = lm.predict(x_test)

    coef = pd.DataFrame(lm.coef_, x.columns, columns=['Coeficientes'])
    print(coef)

    # Quadratic Regression 2
    qpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    qpoly2.fit(x_train, y_train)
    
    # Quadratic Regression 3
    qpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    qpoly3.fit(x_train, y_train)

    # KNN Regression

    ideal = 1
    error_rate = []
    menor = 0
    
    for i in range(1, 51):
        knn = KNeighborsRegressor(n_neighbors=i)
        knn.fit(x_train, y_train)
        predict = knn.predict(x_test)

        if menor < np.mean(predict):
            ideal = i
            menor = np.mean(predict)
        error_rate.append(np.mean(predict))
    
    knn = KNeighborsRegressor(n_neighbors=20)
    knn.fit(x_train, y_train)
    


    # Avaliação
    confidencereg = lm.score(x_test, y_test)
    confidencepoly2 = qpoly2.score(x_test,y_test)
    confidencepoly3 = qpoly3.score(x_test,y_test)
    confidenceknn = knn.score(x_test, y_test)

    print(confidencereg)
    print(confidencepoly2)
    print(confidencepoly3)
    print(confidenceknn)
    print('idea:', ideal)
    print()

    # print(error_rate)


    return ''
    

@app.route('/pricing')
def pricing():
    lm = LinearRegression(n_jobs=-1)
    start = datetime(2015, 1, 1)
    end = datetime(2019, 1, 1)
    stock = web.DataReader('GOOG', 'yahoo', start, end)
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

    confidencereg = clfreg.score(X_test, y_test)
    confidencepoly2 = clfpoly2.score(X_test,y_test)
    confidencepoly3 = clfpoly3.score(X_test,y_test)
    confidenceknn = clfknn.score(X_test, y_test)

    print(confidencereg)
    print(confidencepoly2)
    print(confidencepoly3)
    print(confidenceknn)

    confidence = {'linear_regression': confidencereg, 'quadratic_regression_2': confidencepoly2, 'quadratic_regression_3': confidencepoly3, 'knn_regression': confidenceknn}

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

    dados = {'gráfico': './img/graph.png', 'resultados': confidence}
    return jsonify(dados)



if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)