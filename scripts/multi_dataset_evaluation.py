import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


selected_features = [
"src_bytes",
"dst_bytes",
"flag",
"same_srv_rate",
"dst_host_same_srv_rate",
"diff_srv_rate",
"dst_host_srv_count",
"logged_in",
"dst_host_diff_srv_rate",
"protocol_type",
"dst_host_same_src_port_rate",
"count",
"dst_host_srv_serror_rate",
"service",
"dst_host_srv_diff_host_rate"
]


def preprocess(data):

    categorical = ["protocol_type", "service", "flag"]

    for col in categorical:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    X = data[selected_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


# -------------------------
# NSL-KDD Evaluation
# -------------------------

print("\nEvaluating NSL-KDD")

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

nsl_test = pd.read_csv(
"datasets/raw/NSL_KDD/KDDTest+.txt",
names=columns
)

# assign columns manually if needed
# (same column list used earlier)

# label processing
nsl_test["label"] = nsl_test["label"].astype(str).str.strip()
nsl_test["label"] = nsl_test["label"].apply(lambda x: 0 if x == "normal" else 1)

X_nsl = preprocess(nsl_test)
y_nsl = nsl_test["label"]

model = IsolationForest(n_estimators=200, contamination=0.2)
model.fit(X_nsl)

preds = model.predict(X_nsl)
preds = np.where(preds == -1, 1, 0)

print(classification_report(y_nsl, preds))


# -------------------------
# UNSW-NB15 Evaluation
# -------------------------

print("\nEvaluating UNSW-NB15")

unsw_test = pd.read_parquet(
"datasets/raw/UNSW_NB15/UNSW_NB15_testing-set.parquet"
)

unsw_features = [
"spkts",
"dpkts",
"sbytes",
"dbytes",
"sload",
"dload",
"sinpkt",
"dinpkt"
]

X_unsw = unsw_test[unsw_features]
y_unsw = unsw_test["label"]

scaler = StandardScaler()
X_unsw = scaler.fit_transform(X_unsw)

model = IsolationForest(n_estimators=200, contamination=0.2)
model.fit(X_unsw)

preds = model.predict(X_unsw)
preds = np.where(preds == -1, 1, 0)

print(classification_report(y_unsw, preds))
