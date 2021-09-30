from flask import *
import pandas as pd
import os
from flask_cors import CORS
from flask_jsonpify import jsonpify
import matplotlib.pyplot as plt

app=Flask("__name__")
CORS(app)

@app.route('/upload', methods = ['POST'])  
def upload():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        # print(request.get_data())

        # print("fine")
        try:
            df=pd.read_csv(f.filename)
        except:
            try:
                df=pd.read_excel(f.filename)
            except:
                return jsonify("Please upload a valid file. i.e the file should be CSV or Xlsx")
        print(df)
        return jsonify("File uploaded sucessfully")


@app.route('/<filename>/head')
def head(filename):
    try:
        df=pd.read_csv(filename, header=None)
    except:
        df=pd.read_excel(filename, header=None)

    print("---------------------------------")
  
    head = df.head().values.tolist()
    print(head)
    JSONP_data = jsonpify(head)
    return JSONP_data

@app.route('/<filename>/describe')
def desc(filename):
    try:
        df=pd.read_csv(filename, header=None)
    except:
        df=pd.read_excel(filename, header=None)
    print("---------------------------------")
    desc = df.describe().values.tolist()
    print(desc)
    JSONP_data = jsonpify(desc)
    return JSONP_data

@app.route('/<filename>/plot/<x>/<y>')
def plotgraph (filename,x,y):
    try:
        df=pd.read_csv(filename, header=None)
    except:
        df=pd.read_excel(filename, header=None)
    
    X= df[int(x)].values.tolist()   
    Y= df[int(y)].values.tolist() 
    plt.scatter(X,Y)
    plt.title("distribution")
    plt.xlabel(x)
    plt.ylabel(y)
    print("---------------------------------")
    # print(type(image))
    # return render_template('untitled1.html', name = plt.show())
    plt.savefig("plotimage.png")
    return jsonify("okay")

@app.route('/<filename>/shape')
def shape(filename):
    try:
        df=pd.read_csv(filename, header=None)
    except:
        df=pd.read_excel(filename, header=None)

    print("---------------------------------")

    x,y= df.shape
    print(x,y)
    dictin={"rows":x,"columns":y}
    return jsonify(dictin)


@app.route('/<filename>/<predfile>/linearregnovice')
def linearregnovice(filename,predfile):
    try:
        data=pd.read_csv(filename, header=None)
    except:
        data=pd.read_excel(filename, header=None)
    
    print("---------------------------------")

    try:
        pred=pd.read_csv(predfile, header=None)
    except:
        pred=pd.read_excel(predfile, header=None)
        
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import numpy as np

    X = np.array(data.iloc[:,:-1].values)  
    y = np.array(data.iloc[:,-1].values)
    # X=data.iloc[:,:-1].values # print(len(X)) y=df.iloc[:,-1].values
    
    
    dictin={}
    # Splitting the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
    print("*************************")

    regr = LinearRegression()

    regr.fit(X_train, y_train)
    
    # X_pred=np.array([[8],[6],[5]])
    X_pred = np.array(pred.iloc[:,:].values) 
    # X_pred=np.array([[8,2],[6,5],[5,6]])

    y_pred = regr.predict(X_pred)
    # print(X_test,y_pred)
    print()
    print(regr.score(X_test, y_test))
    dictin["r2_value"]=regr.score(X_test, y_test)
    dictin["X_pred"]=X_pred.tolist()
    dictin["y_pred"]=y_pred.tolist()

    return jsonify(dictin)


if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host= "0.0.0.0", port=port)







