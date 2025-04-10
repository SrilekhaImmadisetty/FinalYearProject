from flask import Flask, request, jsonify, send_file, render_template
from ProjectLatest import predict_and_generate_dag
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        project_name = data.get("project_name")
        task_names = data.get("task_names")
        dep_value = float(data.get("dep_value"))
        indep_value = float(data.get("indep_value"))

        risk_score, risk_level = predict_and_generate_dag(
            project_name, task_names, dep_value, indep_value
        )

        return jsonify({
            "message": f"Risk Score: {risk_score} → {risk_level}",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "dag_image": True
        })
    except Exception as e:
        return jsonify({"error": str(e), "dag_image": False}), 400

@app.route("/get-dag-image")
def get_dag_image():
    image_path = "dag_output.png"
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    return "Image not found", 404

if __name__ == "__main__":
    app.run(debug=True)
