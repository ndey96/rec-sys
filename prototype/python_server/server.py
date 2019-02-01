from flask import Flask

app = Flask(__name__)

@app.route("/getrecd")
def GET():
    return "songIDS"


if __name__ == '__main__':
    app.run(debug=True)