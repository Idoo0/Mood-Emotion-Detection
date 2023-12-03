from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploadFoto'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your pre-trained model (replace 'path/to/your/model.h5' with your actual model path)
model = load_model('model_gambar.h5')

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(48, 48), grayscale=True)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home2')
def home2():
    return render_template('home2.html')

@app.route('/gambar', methods=['GET', 'POST'])
def gambar():
    if request.method == 'POST':
        if 'gambar' in request.files:
            file = request.files['gambar']
            if file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                return redirect(url_for('resultFoto', filename=file.filename))

    return render_template('gambar.html')

@app.route('/suara')
def suara():
    return render_template('suara.html')

@app.route('/resultFoto/<filename>')
def resultFoto(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    predictions = predict_image(image_path)
    categories = ['Sedih', 'Senang', 'Netral', 'Marah']
    predicted_class = np.argmax(predictions)
    predicted_category = categories[predicted_class]
    confidence = predictions[0, predicted_class]
    probability = predictions[0];
    sedih = f"{probability[0]*100:.2f}";
    senang = f"{probability[1]*100:.2f}";
    netral = f"{probability[2]*100:.2f}";
    marah = f"{probability[3]*100:.2f}";
    return render_template('resultFoto.html', filename=filename, predicted_category=predicted_category, confidence=confidence, sedih=sedih, senang=senang, marah=marah, netral=netral)

@app.route('/resultVoice')
def resultVoice():
    return render_template('resultVoice.html')

if __name__ == "__main__":
    app.run(debug=True)
