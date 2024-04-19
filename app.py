from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np

app = Flask(__name__)

__locations = None
__data_columns = None
__model = None

# Function to load saved artifacts
def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns, __locations, __model

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    if __model is None:
        with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

# Function to estimate price based on input
def estimate_price(area, bhk, bathrooms, location):
    loc_index = __data_columns.index(location.lower()) if location.lower() in __data_columns else -1

    x = np.zeros(len(__data_columns))
    x[0] = area
    x[1] = bathrooms
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

@app.route('/')
def home():
    return render_template('app.html')  # Changed to render app.html

@app.route('/estimate_price', methods=['POST'])
def estimate_price_route():
    area = float(request.form['area'])
    bhk = int(request.form['bhk'])
    bathrooms = int(request.form['bathrooms'])
    location = request.form['location']

    estimated_price = estimate_price(area, bhk, bathrooms, location)
    
    return jsonify({'estimated_price': estimated_price})

@app.route('/get_location_names')
def get_location_names():
    return jsonify({'locations': __locations})

@app.route('/get_data_columns')
def get_data_columns():
    return jsonify({'data_columns': __data_columns})

if __name__ == '__main__':
    load_saved_artifacts()
    app.run(debug=True)
