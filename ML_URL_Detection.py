#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[5]:


data = pd.DataFrame({
    "url": [
        "https://www.google.com",
        "https://www.phishing-site.com",
        "https://www.github.com",
        "http://malicious-site.xyz",
        "https://www.yahoo.com",
        "http://fake-login-page.com",
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1 for genuine, 0 for fake
})
print("Class Distribution:\n", data['label'].value_counts())


# In[6]:


# Step 2: Feature extraction
# Use TF-IDF to convert URLs into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['url'])
y = data['label']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[8]:


# Step 4: Train a machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[9]:


# Step 5: Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))


# In[10]:


# Step 6: Predict on a new URL
def predict_url(url):
    url_features = vectorizer.transform([url])
    prediction = model.predict(url_features)
    return "Genuine" if prediction[0] == 1 else "Fake"

# Test the model on new URLs
test_urls = [
    "https://www.google.com",
    "http://phishing-site.com/login",
    "https://www.github.com",
    "http://malicious-site.xyz",
]

for url in test_urls:
    print(f"{url}: {predict_url(url)}")


# In[ ]:




