import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image

app = Flask(__name__)
model = pickle.load(open('inceptionv3_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    path = request.form.get("Image")
    test_image_path = path  # Replace with the actual path to your image
    test_image = image.load_img(test_image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = preprocess_input(test_image)

# Get predictions (class probabilities)
    predictions = model.predict(test_image)
    ind=np.argmax(predictions)
    if ind==0:
        output2="Balck spot"
    elif ind==1:
        output2="Bruise"
    elif ind==2:
        output2="Fresh"
    else:
        output2="Rotten"
        
    output1 = np.max(predictions)*100

    return render_template('index.html', prediction_text = "Class of the given Mango: {} ".format(output2) + "\nPercentage of fit to the class: {}".format(output1))



if __name__ == "__main__":
    app.run(debug=True)