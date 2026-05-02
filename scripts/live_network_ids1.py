import time
import os
import csv
import numpy as np
from sklearn.ensemble import IsolationForest
from scapy.all import sniff, IP, TCP, UDP, ICMP

print("🚀 Hybrid AI IDS Started...\n")

flows = {}
blocked_ips = {}

FLOW_TIMEOUT = 3
MIN_PACKETS = 3
COOLDOWN = 5

WHITELIST = ["127.0.0.1", "192.168.0.1"]

LOG_FILE = "ids_logs.csv"

# -------------------------------
# INIT LOG FILE
# -------------------------------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Source IP", "Packets", "Rate", "Attack"])

# -------------------------------
# 🔥 Isolation Forest Model (LIGHTWEIGHT)
model = IsolationForest(contamination=0.1)

# Train with dummy normal traffic (initialization)
dummy_data = np.array([[1,1,1],[2,1,0],[3,2,1],[1,0,0]])
model.fit(dummy_data)

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

    features = extract_features(flow)
    prediction = model.predict(features)

    return prediction[0] == -1   # anomaly

# -------------------------------
def rule_detect(flow):

    duration = max(flow["last"] - flow["start"], 0.1)
    rate = flow["packets"] / duration

    if flow["syn"] > 8:
        return "SYN Flood"

    if len(flow["ports"]) > 10:
        return "Port Scan"

    if flow["icmp"] > 10:
        return "ICMP Flood"

    if rate > 20:
        return "DDoS"

    return "Normal"

# -------------------------------
def hybrid_decision(flow):

    src = flow["src"]

    if src in WHITELIST:
        return "Normal"

    ai_flag = ai_detect(flow)
    rule_result = rule_detect(flow)

    # 🔥 HYBRID LOGIC
    if ai_flag and rule_result != "Normal":
        return rule_result + " (AI Confirmed)"

    if ai_flag:
        return "Anomaly (AI)"

    return rule_result

# -------------------------------
def log_attack(flow, attack):

    duration = max(flow["last"] - flow["start"], 0.1)
    rate = flow["packets"] / duration

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime("%H:%M:%S"),
            flow["src"],
            flow["packets"],
            round(rate, 2),
            attack
        ])

# -------------------------------
def block_ip(ip):
    now = time.time()

    if ip in blocked_ips and now - blocked_ips[ip] < COOLDOWN:
        return

    os.system(f"iptables -A INPUT -s {ip} -j DROP")
    blocked_ips[ip] = now

    print(f"🚫 BLOCKED: {ip}")

# -------------------------------
def process_packet(packet):

    if IP not in packet:
        return

    src = packet[IP].src

    if src not in flows:
        flows[src] = create_flow(packet)

    update_flow(flows[src], packet)

    if flows[src]["packets"] >= MIN_PACKETS:

        attack = hybrid_decision(flows[src])

        duration = max(flows[src]["last"] - flows[src]["start"], 0.1)
        rate = flows[src]["packets"] / duration

        print(f"[{src}] rate={round(rate,1)} -> {attack}")

        if attack != "Normal":
            print(f"🚨 ALERT: {attack}")
            log_attack(flows[src], attack)
            block_ip(src)

        flows[src] = create_flow(packet)

# -------------------------------
sniff(prn=process_packet, store=0, iface=["eth0", "lo"])