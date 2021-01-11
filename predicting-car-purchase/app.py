import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('car_sales_predict.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features =np.array(int_features).reshape(1,5)
    prediction = model.predict(final_features)

    output = prediction[0][0]

    return render_template('index.html', prediction_text='The customer can be expected to purchase around $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)