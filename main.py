from flask import Flask , render_template,request
from app.utils import Wine_Prediction
import CONFIG

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def load_raw():
    data = request.form
    pred_obj = Wine_Prediction()
    predict_quality = pred_obj.predict_wine_quality(data)
    print(predict_quality)

    return render_template("index.html",qua = predict_quality)


if __name__=="__main__":
    app.run(host=CONFIG.HOST,port = CONFIG.PORT, debug= CONFIG.DEBUG)