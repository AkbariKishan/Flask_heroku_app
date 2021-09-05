from flask import Flask, render_template, request
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def house_price_pred():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html')
    else:
        int_features = np.array([float(x) for x in request.form.values() if is_float(x)])
        if int_features.shape[0] == 15:
            int_features_final = int_features.reshape(1,-1)
            model = load('model.joblib')
            scalery = load(open('scaler_y.pkl', 'rb'))
            scalerX = load(open('scaler_X.pkl', 'rb'))
            results = make_pred(model, scalerX, scalery, int_features_final)
        else:
            results = "Non-float value(s) inserted! Please insert values as per insructions."
        return render_template('index.html', prediction_text=results)

def make_pred(model, scaler_X, scaler_y, new_array):
    scaled_X = scaler_X.transform(new_array)
    y_pred = model.predict(scaled_X)
    y_pred_inv_scaled = scaler_y.inverse_transform(y_pred.reshape(-1,1))
    return y_pred_inv_scaled

def is_float(s):
    try:
        float(s)
        return True
    except:
        return False


if __name__ == "__main__":
    app.run(debug=True)
