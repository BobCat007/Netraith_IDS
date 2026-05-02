import time
import pandas as pd
import joblib

print("Starting Real-Time IDS...")

model = joblib.load("models/attack_classifier_xgb.pkl")

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

df = pd.read_csv(
"datasets/raw/CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
)

df.columns = df.columns.str.strip()

X = df[features]

for i in range(20):

    sample = X.sample(1)

    pred = model.predict(sample)[0]

    if pred != 0:
        print("⚠ ATTACK DETECTED")

    else:
        print("Normal traffic")

    time.sleep(1)
