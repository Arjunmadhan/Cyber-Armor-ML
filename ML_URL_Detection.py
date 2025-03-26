import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.DataFrame({
    "url": [
        "https://www.google.com",
        "https://www.phishing-site.com",
        "https://www.github.com",
        "http://malicious-site.xyz",
        "https://www.yahoo.com",
        "http://fake-login-page.com",
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1 = Genuine, 0 = Fake
})


def extract_features(url):
    return {
        "url_length": len(url),
        "num_dots": url.count('.'),
        "num_slashes": url.count('/'),
        "num_digits": sum(c.isdigit() for c in url),
        "contains_https": 1 if url.startswith("https") else 0,
        "num_special_chars": len(re.findall(r'[\W_]', url)),  # Special characters count
        "contains_login": 1 if "login" in url.lower() else 0,
        "contains_secure": 1 if "secure" in url.lower() else 0
    }

# Convert URLs into structured numerical features
features = pd.DataFrame([extract_features(url) for url in data["url"]])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Function to predict new URLs
def predict_url(url):
    url_features = pd.DataFrame([extract_features(url)])
    prediction = model.predict(url_features)
    return "Genuine" if prediction[0] == 1 else "Fake"

# Test the model with new URLs
test_urls = [
    "https://www.google.com",
    "http://phishing-site.com/login",
    "https://www.github.com",
    "http://malicious-site.xyz",
]

for url in test_urls:
    print(f"{url}: {predict_url(url)}")






