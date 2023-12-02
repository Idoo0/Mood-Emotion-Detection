from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home2')
def home2():
    return render_template('home2.html')

@app.route('/gambar')
def gambar():
    return render_template('gambar.html')

@app.route('/suara')
def suara():
    return render_template('suara.html')

@app.route('/resultpic')
def resultFoto():
    return render_template('resultFoto.html')

@app.route('/resultVoice')
def resultVoice():
    return render_template('resultVoice.html')

if __name__=="__main__":
    app.run()