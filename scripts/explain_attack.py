import shap
import joblib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Prevent display issues in WSL
matplotlib.use("Agg")

print("Loading model...")

model = joblib.load("models/attack_classifier_xgb.pkl")

explainer = shap.TreeExplainer(model)

df = pd.read_csv(
"datasets/raw/CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
)

df.columns = df.columns.str.strip()

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

X = df[features].iloc[:100]

print("Generating SHAP explanations...")

shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, show=False)

plt.savefig("shap_explanation.png")

print("SHAP explanation saved as shap_explanation.png")
