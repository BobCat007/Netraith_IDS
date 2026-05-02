import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

test_path = "/mnt/d/ai_ids_project/datasets/raw/NSL_KDD/KDDTest+.txt"

columns = [
"duration","protocol_type","service","flag","src_bytes","dst_bytes",
"land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
"num_compromised","root_shell","su_attempted","num_root","num_file_creations",
"num_shells","num_access_files","num_outbound_cmds","is_host_login",
"is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
"rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
"srv_diff_host_rate","dst_host_count","dst_host_srv_count",
"dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
"dst_host_serror_rate","dst_host_srv_serror_rate",
"dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

data = pd.read_csv(test_path, names=columns)

data["label"] = data["label"].astype(str).str.strip()
data["label"] = data["label"].apply(lambda x: 0 if x == "normal" else 1)

categorical = ["protocol_type","service","flag"]

for col in categorical:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data.drop(["label","difficulty"], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(n_estimators=200, contamination=0.2, random_state=42)
model.fit(X_scaled)

explainer = shap.TreeExplainer(model)

sample = X_scaled[:200]

shap_values = explainer.shap_values(sample)

shap.summary_plot(shap_values, sample, feature_names=X.columns, show=False)

plt.savefig("shap_feature_importance.png", dpi=300, bbox_inches="tight")

print("SHAP explanation saved as shap_feature_importance.png")
