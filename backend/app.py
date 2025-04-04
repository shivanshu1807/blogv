import joblib
import pandas as pd
import os
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("datasets", exist_ok=True)

# Include C language
LANGUAGES = ["python", "javascript", "java", "cpp", "c"]

# Function to generate random code-like text
def generate_random_code(language, num_samples=500):
    snippets = []
    for _ in range(num_samples):
        # Generate a pseudo code snippet
        code_length = random.randint(20, 200)
        code = ''.join(random.choices(string.ascii_letters + string.digits + " (){}[]=;<>!+-*/", k=code_length))

        # Assign a random technical debt score (0.0 to 1.0)
        debt_score = round(random.uniform(0, 1), 2)

        snippets.append([code, debt_score])

    return pd.DataFrame(snippets, columns=["code", "debt_score"])

# Generate datasets and save them as CSV
for lang in LANGUAGES:
    dataset_path = f"datasets/{lang}_tech_debt.csv"
    df = generate_random_code(lang)
    df.to_csv(dataset_path, index=False)
    print(f"✅ Generated dataset: {dataset_path}")

# Function to train and save model
def train_and_save_model(language, dataset_path):
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found for {language}: {dataset_path}")
        return

    # Load dataset
    df = pd.read_csv(dataset_path)

    if "code" not in df or "debt_score" not in df:
        print(f"❌ Dataset for {language} must have 'code' and 'debt_score' columns")
        return

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df["code"], df["debt_score"], test_size=0.2, random_state=42)

    # Convert source code into TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_tfidf)
    error = mean_absolute_error(y_test, y_pred)
    print(f"📊 {language.upper()} Model MAE: {error:.4f}")

    # Save model & vectorizer together
    model_path = f"models/{language}_tech_debt.pkl"
    
    
    joblib.dump({"model": model, "vectorizer": vectorizer}, model_path)
    print(f"💾 Saved model: {model_path}")

# Train models for each language
for lang in LANGUAGES:
    train_and_save_model(lang, f"datasets/{lang}_tech_debt.csv")
    
    
    
    
    
    
    
    
    
    
    