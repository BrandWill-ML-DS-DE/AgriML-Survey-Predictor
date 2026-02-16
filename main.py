import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. LOAD YOUR DATASET
url = "https://raw.githubusercontent.com/Gladiator07/Harvestify/refs/heads/master/Data-processed/crop_recommendation.csv"
df = pd.read_csv(url)

# 2. SYNTHESIZE SURVEY METRICS
np.random.seed(42)
df['Farm_Size_Acres'] = np.random.gamma(shape=2, scale=15, size=len(df))
df['Education_Level'] = np.random.choice([1, 2, 3, 4, 5], len(df)) # 1: No High School to 5: Graduate

# Simulate a "Likert Scale" question: "How much do you trust new agroforestry tech?" (1-5)
df['Trust_Score_Likert'] = np.random.choice([1, 2, 3, 4, 5], len(df), p=[0.1, 0.2, 0.4, 0.2, 0.1])

# 3. PROPENSITY WEIGHTING (Addressing "Survey Nonresponse")
df['Responded'] = df.apply(lambda x: 0 if (x['Farm_Size_Acres'] < 10 and np.random.rand() < 0.5) else 1, axis=1)

# Use Random Forest to calculate the Propensity Score (Probability of responding)
X_prop = df[['Farm_Size_Acres', 'Education_Level', 'ph']]
y_prop = df['Responded']
prop_model = RandomForestClassifier(n_estimators=100).fit(X_prop, y_prop)
df['Propensity_Score'] = prop_model.predict_proba(X_prop)[:, 1]
df['Weight'] = 1 / (df['Propensity_Score'] + 0.01)

# 4. NLP SENTIMENT ANALYSIS SIMULATION
feedbacks = ["Highly interested in diversification", "Too expensive to implement", "Need more technical training", "Worried about soil acidity"]
df['Qualitative_Feedback'] = [np.random.choice(feedbacks) for _ in range(len(df))]
df['Sentiment_Feature'] = df['Qualitative_Feedback'].apply(lambda x: 1 if "interested" in x or "Great" in x else 0)

# 5. THE PREDICTIVE MODEL (Hybrid Data)
# Target: Adoption Willingness (Binary)
df['Adoption_Willingness'] = ((df['rainfall'] > 100) & (df['Trust_Score_Likert'] >= 3)).astype(int)

# Filter for the 'Respondents' (Simulating real-world survey analysis)
respondents = df[df['Responded'] == 1].copy()

X = respondents[['N', 'P', 'K', 'ph', 'rainfall', 'Farm_Size_Acres', 'Trust_Score_Likert', 'Sentiment_Feature']]
y = respondents['Adoption_Willingness']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

print("Project: AgriML-Survey-Predictor Implementation Complete.")
print(classification_report(y_test, clf.predict(X_test)))
