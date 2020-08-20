from flask import Flask, jsonify, request, render_template,url_for,jsonify
from  models.utils import predict

app = Flask(__name__)


@app.route("/",methods=["GET","POST"])
def index():
    vals=["Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]
    if request.method=="POST":
        ip=[]
        for i in vals:
            ip.append(int(request.form.get(i)))
        prediction=predict(ip)
        predicts={
            "output":"diabetic" if prediction else "you can eat more sweets",
            "inputs":{
                "Glucose":ip[0],
                "BloodPressure":ip[1],
                "SkinThickness":ip[2],
                "Insulin":ip[3],
                "BMI":ip[4],
                "Age":ip[5],
            }
            
            }
        return jsonify(predicts)
    return render_template("index.htm",c=vals)


if __name__ == "__main__":
    app.run(debug=True)
