from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get all emails from the form
    messages = request.form.getlist("message")

    results = []

    for msg in messages:

        data = vectorizer.transform([msg])

        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "🚨 Spam Email"
        else:
            result = "✅ Ham Email"

        results.append({
            "message": msg,
            "prediction": result
        })

    return render_template(
        "index.html",
        results=results
    )


if __name__ == "__main__":
    app.run(debug=True)