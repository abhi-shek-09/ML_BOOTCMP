from flask import Flask, request

from movierecfinal import *

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        result=get_recommendations(data['text']) # 'text'='The Dark Knight Rises'
        return result

if __name__=="__main__":
    app.run(debug=True)
