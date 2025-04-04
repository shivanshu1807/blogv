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
    
    import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
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
        print(f"✅ Loaded model for {lang}")
    else:
        print(f"❌ Model not found for {lang}: {path}")


def calculate_metrics(code):
    """Calculates LOC, Maintainability Index, Bug Density, and Comment Density."""
    lines = code.split("\n")
    loc = len(lines)
   
    mi = max(0, min(100, 171 - 5.2 * np.log1p(loc) - 0.23 * (len(code) / max(1, loc)) * 100))
    bug_density = round((0.05 * loc) / max(1, loc), 3)
    comment_lines = sum(1 for line in lines if line.strip().startswith(("#", "//", "/*", "*", "--")))
    comment_density = round(comment_lines / max(1, loc), 3)
   
    return loc, mi, bug_density, comment_density


def generate_graphs(loc, mi, bug_density, comment_density):
    """Generates a line graph for the metrics and returns base64 encoded images."""
    metrics = ["LOC", "Maintainability Index", "Bug Density", "Comment Density"]
    values = [loc, mi, bug_density, comment_density]
   
    plt.figure(figsize=(6, 4))
    plt.plot(metrics, values, marker="o", linestyle="-", color="blue")
    plt.title("Code Quality Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.grid(True)
   
    img_io = io.BytesIO()
    plt.savefig(img_io, format="png")
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode("utf-8")
    plt.close()
   
    return img_base64


def analyze_technical_debt(language, code):
    if language in models:
        model_data = models[language]
        model, vectorizer = model_data["model"], model_data["vectorizer"]
       
        code_vectorized = vectorizer.transform([code])
        prediction = model.predict(code_vectorized)[0]
        risk = "High" if prediction > 0.7 else "Medium" if prediction > 0.4 else "Low"
       
        loc, mi, bug_density, comment_density = calculate_metrics(code)
        graph_base64 = generate_graphs(loc, mi, bug_density, comment_density)
       
        return {
            "technical_debt_score": round(prediction, 2),
            "risk": risk,
            "language": language,
            "lines_of_code": loc,
            "maintainability_index": round(mi, 2),
            "bug_density": bug_density,
            "comment_density": comment_density,
            "metrics_graph": graph_base64
        }
   
    return {"technical_debt_score": "N/A", "message": "No model found for this language"}


@app.route('/predict', methods=['POST'])
def predict_technical_debt():
    try:
        data = request.json
        print(f"Received Request: {data}")

        repo = data.get("repo")
        commit = data.get("commit")
        files = data.get("files", {})

        if not files:
            return jsonify({"error": "No files received"}), 400

        results = {}
        for filename, content in files.items():
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
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
    
    
    ne