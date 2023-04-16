import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from keras.applications import VGG16
from tensorflow.keras.preprocessing import image as Image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Define the Flask app
app = Flask(__name__)

# Set the allowed file types for upload
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


# Define a function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Define the route for the homepage
@app.route('/')
def home():
    return render_template('home.html')


# Define the route for the image upload
@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Save the uploaded file
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Load the image and preprocess it for the VGG16 model
        img = Image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(224, 224))
        x = Image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Use the pre-trained model to make a prediction on the image
        preds = model.predict(x)
        label = decode_predictions(preds, top=1)[0][0][1]

        #delete the file after prediction
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Return the predicted label
        return render_template('result.html', label=label)
    
        

    # If the file is not valid, redirect to the homepage
    else:
        return redirect(request.url)


# Define the route for the image upload with VGG16 predictions
@app.route('/vgg16_upload', methods=['POST'])
def vgg16_upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img = Image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(224, 224))
    img = Image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    prediction = model.predict(img)
    decoded_prediction = decode_predictions(prediction, top=1)[0][0]
    result = {"object": decoded_prediction[1], "probability": str(decoded_prediction[2])}
    return render_template('result.html', result=result)


# Define the configuration for the Flask app
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'secret_key'

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)