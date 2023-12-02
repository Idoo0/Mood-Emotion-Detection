from flask import Flask, render_template

app = Flask(__name__)


@app.route('/gambar')
def gambar():
    return "Hello"

if __name__ == "__main__":
    app.run(debug=True)