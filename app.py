from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home2():
    return render_template('home2.html')

@app.route('/gambar')
def gambar():
    return render_template('gambar.html')

if __name__=="__main__":
    app.run()