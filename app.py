from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploadFoto'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home2')
def home2():
    return render_template('home2.html')

@app.route('/gambar', methods=['GET', 'POST'])
def gambar():
    if request.method == 'POST':
        # Memastikan bahwa 'gambar' adalah nama field pada form
        if 'gambar' in request.files:
            file = request.files['gambar']
            if file.filename != '':
                # Menggabungkan path upload dengan nama file
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                # Menyimpan file ke folder uploadFoto
                file.save(filepath)
                # Redirect ke halaman resultFoto.html atau halaman lain yang diinginkan
                return redirect(url_for('resultFoto', filename=file.filename))

    return render_template('gambar.html')

@app.route('/suara')
def suara():
    return render_template('suara.html')

@app.route('/resultFoto')
def resultFoto():
    return render_template('resultFoto.html')

@app.route('/resultVoice')
def resultVoice():
    return render_template('resultVoice.html')

if __name__ == "__main__":
    app.run()
