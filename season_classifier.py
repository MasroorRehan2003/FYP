import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, jaccard_score
import joblib

# === Load and preprocess dataset ===
df = pd.read_csv(r"C:\Users\hP\Desktop\FYP\FYP_DATASET_Final\Cleaned_Final_csv.csv")
df["gender"] = df["gender"].str.lower()
df["season"] = df["season"].str.lower()
df["articleType"] = df["articleType"].str.lower()

# === Group by articleType and aggregate seasons ===
df_grouped = df.groupby("articleType")["season"].apply(list).reset_index()

# === Features and labels ===
X = df_grouped["articleType"]
y = df_grouped["season"]

# === Binarize the labels ===
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y)

# === Train/test split ===
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# === Build pipeline ===
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42)))
])

# === Train model ===
pipeline.fit(X_train, Y_train)

# === Evaluate model ===
Y_pred = pipeline.predict(X_test)
print(f"\n✅ Hamming Loss: {hamming_loss(Y_test, Y_pred):.2f}")

print("📊 Classification Report:\n")
print(classification_report(Y_test, Y_pred, target_names=mlb.classes_))

# === Save the model and binarizer ===
joblib.dump(pipeline, "season_multilabel_model_CLEANNNNNNN_2.joblib")
joblib.dump(mlb, "season_mlb_2.joblib")
print("\n✅ Model and label binarizer saved.")


