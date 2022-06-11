from flask import Flask, jsonify, render_template, request
import os
import numpy
import pickle
from sklearn.ensemble import RandomForestRegressor


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        Item_Weight = float(request.form['Item_Weight'])
        Item_Fat_Content = float(request.form['Item_Fat_Content'])
        Item_Visibility = float(request.form['Item_Visibility'])
        Item_Type = float(request.form['Item_Type'])
        Item_MRP = float(request.form['Item_MRP'])
        Outlet_Size = float(request.form['Outlet_Size'])
        Outlet_Location_Type = float(request.form['Outlet_Location_Type'])
        Outlet_Type = float(request.form['Outlet_Type'])
        Years_Established = float(request.form['Years_Established'])

        data = [[Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP
                    , Outlet_Size, Outlet_Location_Type, Outlet_Type, Years_Established]]

        rf = pickle.load(open('rf.pkl', 'rb'))

        prediction = rf.predict(data)
    return render_template('home.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True, port=3030)
