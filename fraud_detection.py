import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Step 1: Load datasets
# -----------------------------
train_file = "fraudTrain.csv"
test_file = "fraudTest.csv"

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

print("✅ Datasets loaded successfully!")
print("Training set shape:", df_train.shape)
print("Test set shape:", df_test.shape)

# -----------------------------
# Step 2: Use smaller subset for faster training
# -----------------------------
df_train_small = df_train.sample(50000, random_state=42)
cols = ['unix_time', 'amt']

X_train = df_train_small[cols]
y_train = df_train_small['is_fraud']

X_test = df_test[cols]
y_test = df_test['is_fraud']

# -----------------------------
# Step 3: Scale features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Step 4: Train Random Forest
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

print("\n🎯 Fraud Detection Model Ready!")
print("Enter transactions to predict (type 'exit' to quit)")

# -----------------------------
# Step 5: User input for prediction
# -----------------------------
while True:
    user_input = input("\nEnter unix_time, amt (e.g., 1325376018, 123.45):\n")
    if user_input.lower() == "exit":
        break
    try:
        unix_time_val, amt_val = [float(x.strip())
                                  for x in user_input.split(",")]
        # Convert input to DataFrame with same column names to avoid warning
        input_df = pd.DataFrame(
            [[unix_time_val, amt_val]], columns=['unix_time', 'amt'])
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0, 1]  # probability of fraud
        label = "Fraudulent" if prob >= 0.5 else "Legitimate"
        print(f"✅ Predicted: {label} (Fraud Probability: {prob:.4f})")
    except Exception as e:
        print("❌ Error:", e)
        print("Format: unix_time, amt  e.g., 1325376018, 123.45")

# -----------------------------
# Optional: Show top 20 suspicious transactions from test set
# -----------------------------
p_fraud_test = model.predict_proba(X_test_scaled)[:, 1]
top20 = df_test.copy()
top20['p_fraud'] = p_fraud_test
print("\n🔝 Top 20 suspicious transactions in test set:")
print(top20.sort_values('p_fraud', ascending=False).head(
    20)[['unix_time', 'amt', 'p_fraud', 'is_fraud']])
