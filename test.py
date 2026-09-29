# Step 1: Import the tools from Scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 2: Our training data — example safety reports
# These are the "examples" we show the computer
reports = [
    "worker on ladder with no harness",
    "gas leak detected near valve",
    "safety meeting completed on time",
    "unguarded machinery near walkway",
    "all equipment checked and fine",
    "worker slipped but no injury",
    "forklift speeding ear workers",    #NEW - dangerous
    "forklift parked safety in bay",    #NEW - safe
]

# Step 3: The correct answers (labels) for each report
# 1 = dangerous (SIF precursor), 0 = safe
labels = [1, 1, 0, 1, 0, 1,1,0]

# Step 4: Convert text to numbers (computers can't read words, only numbers)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reports)

# Step 5: Train the model — this is where "learning" happens
model = MultinomialNB()
model.fit(X, labels)

# Step 6: Test it on a NEW report it has never seen
new_report = ["worker on ladder with no harness"]
new_X = vectorizer.transform(new_report)
prediction = model.predict(new_X)

# Step 7: Show the result
if prediction[0] == 1:
    print("⚠️  DANGEROUS — This looks like a SIF precursor!")
else:
    print("✅ SAFE — No immediate concern.")
import joblib
joblib.dump(model,'model.pkl')
joblib.dump(vectorizer,'vectorizer.pkl')