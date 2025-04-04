import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Model paths
MODEL_PATHS = {
    "python": "models/python_tech_debt.pkl",
    "javascript": "models/javascript_tech_debt.pkl",
    "java": "models/java_tech_debt.pkl",
    "cpp": "models/cpp_tech_debt.pkl"
}

# Load models at startup
models = {}
for lang, path in MODEL_PATHS.items():
    if os.path.exists(path):
        models[lang] = joblib.load(path)
        print(f"Loaded model for {lang}")  # Debugging line
    else:
        print(f"Model not found for {lang}: {path}")  # Debugging line

# Function to analyze technical debt
def analyze_technical_debt(language, code):
    if language in models:
        model_data = models[language]
        model, vectorizer = model_data["model"], model_data["vectorizer"]
        
        # Transform input code
        code_vectorized = vectorizer.transform([code])
        
        # Predict technical debt score
        prediction = model.predict(code_vectorized)[0]
        
        # Assign risk level
        if prediction > 0.7:
            risk = "High"
        elif prediction > 0.4:
            risk = "Medium"
        else:
            risk = "Low"

        return {"technical_debt_score": round(prediction, 2), "risk": risk, "language": language}
    
    return {"technical_debt_score": "N/A", "message": "No model found for this language"}

# API route to predict technical debt
@app.route('/predict', methods=['POST'])
def predict_technical_debt():
    try:
        data = request.json
        print(f"Received Request: {data}")  # Debugging line

        repo = data.get("repo")
        commit = data.get("commit")
        files = data.get("files", {})

        if not files:
            return jsonify({"error": "No files received"}), 400

        results = {}
        for filename, content in files.items():
            # Determine language based on file extension
            if filename.endswith(".py"):
                language = "python"
            elif filename.endswith(".js"):
                language = "javascript"
            elif filename.endswith(".java"):
                language = "java"
            elif filename.endswith(".cpp") or filename.endswith(".c"):
                language = "cpp"
            else:
                language = "unknown"
            
            results[filename] = analyze_technical_debt(language, content)

        return jsonify({"repository": repo, "commit": commit, "analysis": results})

    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging line
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)