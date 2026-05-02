import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import glob

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

print("Loading CICIDS dataset...")

files = glob.glob("datasets/raw/CICIDS2017/MachineLearningCVE/*.csv")

dfs = []

for file in files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

features = [
"Fwd Packet Length Max",
"Avg Fwd Segment Size",
"Total Length of Fwd Packets",
"Fwd IAT Max",
"Fwd IAT Std",
"Fwd Packet Length Mean",
"Fwd IAT Total",
"Init_Win_bytes_forward",
"Subflow Fwd Bytes",
"Fwd IAT Mean",
"Total Fwd Packets",
"Fwd Header Length.1",
"act_data_pkt_fwd",
"Destination Port",
"Fwd Packet Length Std"
]

X = df[features]
y = df["Label"]

print("Loading trained attack classifier...")

model = joblib.load("models/attack_classifier_xgb.pkl")

preds = model.predict(X)

# -------------------------
# Confusion Matrix
# -------------------------

cm = confusion_matrix(y, preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig("results_confusion_matrix.png")

plt.close()

# -------------------------
# ROC Curve
# -------------------------

probs = model.predict_proba(X)[:,1]

fpr, tpr, _ = roc_curve(y, probs)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.2f)" % roc_auc)

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("results_roc_curve.png")

plt.close()

# -------------------------
# Precision Recall Curve
# -------------------------

precision, recall, _ = precision_recall_curve(y, probs)

plt.figure()

plt.plot(recall, precision)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.savefig("results_precision_recall_curve.png")

plt.close()

# -------------------------
# Feature Importance
# -------------------------

importance = model.feature_importances_

feature_imp = pd.Series(importance, index=features)

feature_imp = feature_imp.sort_values(ascending=False)

plt.figure(figsize=(10,6))

feature_imp.plot(kind='bar')

plt.title("Feature Importance")

plt.savefig("results_feature_importance.png")

plt.close()

# -------------------------
# Attack Distribution
# -------------------------

attack_counts = df["Label"].value_counts()

plt.figure()

attack_counts.plot(kind="bar")

plt.title("Attack vs Normal Distribution")

plt.savefig("results_attack_distribution.png")

plt.close()

print("All graphs generated successfully.")
