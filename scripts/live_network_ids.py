import json
import time
import os
import csv
import numpy as np
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from scapy.all import sniff, IP, TCP, UDP, ICMP

print("🚀 HYBRID IDS STARTED...\n")

flows = {}
attacker_counts = defaultdict(int)
logs_list = []

FLOW_TIMEOUT = 3
MIN_PACKETS = 4
WHITELIST = ["127.0.0.1", "192.168.0.1"]

LOG_FILE = "ids_logs.csv"
DATA_FILE = "live_data.json"

# -------------------------------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Source IP", "Packets", "Rate", "Attack"])

# -------------------------------
model = IsolationForest(contamination=0.1)

train_data = np.array([
    [1,1,0],
    [2,1,0],
    [3,2,1],
    [1,0,0],
    [2,1,1]
])

model.fit(train_data)

baseline_rate = 5

# -------------------------------
def update_baseline(flow):
    global baseline_rate
    duration = max(flow["last"] - flow["start"], 0.1)
    rate = flow["packets"] / duration
    baseline_rate = 0.9 * baseline_rate + 0.1 * rate

# -------------------------------
def create_flow(packet):
    return {
        "start": time.time(),
        "last": time.time(),
        "packets": 0,
        "syn": 0,
        "ack": 0,
        "icmp": 0,
        "src": packet[IP].src,
        "ports": set()
    }

# -------------------------------
def update_flow(flow, packet):
    flow["packets"] += 1
    flow["last"] = time.time()

    if TCP in packet:
        flow["ports"].add(packet[TCP].dport)

        if packet[TCP].flags == "S":
            flow["syn"] += 1
        elif packet[TCP].flags == "A":
            flow["ack"] += 1

    if ICMP in packet:
        flow["icmp"] += 1

# -------------------------------
def extract_features(flow):
    duration = max(flow["last"] - flow["start"], 0.1)
    rate = flow["packets"] / duration
    return np.array([[flow["packets"], rate, flow["syn"]]])

# -------------------------------
def ai_detect(flow):
    return model.predict(extract_features(flow))[0] == -1

# -------------------------------
def rule_detect(flow):

    duration = max(flow["last"] - flow["start"], 0.1)
    rate = flow["packets"] / duration

    if flow["packets"] < MIN_PACKETS or rate < 5:
        return "Normal"

    if flow["syn"] > 8:
        return "SYN Flood"

    if len(flow["ports"]) > 5:
        return "Port Scan"

    if flow["icmp"] > 10:
        return "ICMP Flood"

    if rate > baseline_rate * 2:
        return "DDoS"

    return "Normal"

# -------------------------------
def hybrid_decision(flow):

    if flow["src"] in WHITELIST:
        return "Normal"

    ai_flag = ai_detect(flow)
    rule = rule_detect(flow)

    duration = max(flow["last"] - flow["start"], 0.1)
    rate = flow["packets"] / duration

    if rate < 5:
        return "Normal"

    if ai_flag and rule != "Normal":
        return rule

    if ai_flag and rate > 8:
        return "Anomaly"

    return rule

# -------------------------------
def log_attack(flow, attack):

    duration = max(flow["last"] - flow["start"], 0.1)
    rate = flow["packets"] / duration

    log_entry = {
        "Time": time.strftime("%H:%M:%S"),
        "Source IP": flow["src"],
        "Packets": flow["packets"],
        "Rate": round(rate, 2),
        "Attack": attack
    }

    logs_list.append(log_entry)
    attacker_counts[flow["src"]] += 1

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            log_entry["Time"],
            log_entry["Source IP"],
            log_entry["Packets"],
            log_entry["Rate"],
            log_entry["Attack"]
        ])

# -------------------------------
def update_dashboard():

    data = {
        "total": sum(attacker_counts.values()),
        "attackers": len(attacker_counts),
        "logs": logs_list[-10:],   # last 10 logs
        "top": dict(attacker_counts)
    }

    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

# -------------------------------
def process_packet(packet):

    if IP not in packet:
        return

    src = packet[IP].src

    if src not in flows:
        flows[src] = create_flow(packet)

    update_flow(flows[src], packet)
    update_baseline(flows[src])

    if flows[src]["packets"] >= MIN_PACKETS:

        attack = hybrid_decision(flows[src])

        duration = max(flows[src]["last"] - flows[src]["start"], 0.1)
        rate = flows[src]["packets"] / duration

        print(f"[{src}] rate={round(rate,1)} → {attack}")

        if attack != "Normal":
            print(f"🚨 ALERT: {attack}")
            log_attack(flows[src], attack)

        update_dashboard()   # 🔥 SEND DATA TO DASHBOARD

        flows[src] = create_flow(packet)

# -------------------------------
sniff(prn=process_packet, store=0, iface=["eth0","lo"])