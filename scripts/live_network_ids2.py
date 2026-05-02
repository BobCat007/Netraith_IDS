import time
import os
import csv
import numpy as np
from sklearn.ensemble import IsolationForest
from scapy.all import sniff, IP, TCP, UDP, ICMP

print("🚀 Ultimate Adaptive Hybrid IDS Started...\n")

# -----------------------------------
flows = {}
blocked_ips = {}

FLOW_TIMEOUT = 3
MIN_PACKETS = 3
COOLDOWN = 5
BLOCK_TIME = 60

WHITELIST = ["127.0.0.1", "192.168.0.1"]

LOG_FILE = "ids_logs.csv"

# -----------------------------------
# INIT LOG FILE
# -----------------------------------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Source IP", "Packets", "Rate", "Attack"])

# -----------------------------------
# GAN-LIKE SYNTHETIC GENERATOR
# -----------------------------------
def generate_synthetic_attack():
    packets = np.random.randint(20, 50)
    rate = np.random.uniform(20, 50)
    syn = np.random.randint(10, 30)
    return np.array([packets, rate, syn])

# -----------------------------------
# AI MODEL
# -----------------------------------
model = IsolationForest(contamination=0.1)

normal_data = np.array([[1,1,1],[2,1,0],[3,2,1],[1,0,0]])
synthetic_data = np.array([generate_synthetic_attack() for _ in range(50)])

train_data = np.vstack((normal_data, synthetic_data))
model.fit(train_data)

print("✔ GAN-inspired training applied\n")

# -----------------------------------
# ADAPTIVE BASELINE
# -----------------------------------
baseline_rate = 5

def update_baseline(flow):
    global baseline_rate
    duration = max(flow["last"] - flow["start"], 0.1)
    rate = flow["packets"] / duration
    baseline_rate = 0.9 * baseline_rate + 0.1 * rate

# -----------------------------------
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

# -----------------------------------
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

# -----------------------------------
def extract_features(flow):
    duration = max(flow["last"] - flow["start"], 0.1)
    rate = flow["packets"] / duration
    return np.array([[flow["packets"], rate, flow["syn"]]])

# -----------------------------------
def ai_detect(flow):
    return model.predict(extract_features(flow))[0] == -1

# -----------------------------------
def rule_detect(flow):
    duration = max(flow["last"] - flow["start"], 0.1)
    rate = flow["packets"] / duration

    dynamic_threshold = baseline_rate * 2

    if flow["syn"] > 8:
        return "SYN Flood"

    if len(flow["ports"]) > 10:
        return "Port Scan"

    if flow["icmp"] > 10:
        return "ICMP Flood"

    if rate > dynamic_threshold:
        return "DDoS"

    return "Normal"

# -----------------------------------
def explain(flow, ai_flag):

    duration = max(flow["last"] - flow["start"], 0.1)
    rate = flow["packets"] / duration
    threshold = baseline_rate * 2

    reasons = []

    if rate > threshold:
        reasons.append("High packet rate")

    if flow["syn"] > 8:
        reasons.append("Excess SYN packets")

    if len(flow["ports"]) > 10:
        reasons.append("Multiple ports scanned")

    if flow["icmp"] > 10:
        reasons.append("High ICMP traffic")

    if ai_flag:
        reasons.append("AI anomaly detected")

    return reasons

# -----------------------------------
def hybrid_decision(flow):

    if flow["src"] in WHITELIST:
        return "Normal", False

    ai_flag = ai_detect(flow)
    rule_result = rule_detect(flow)

    if ai_flag and rule_result != "Normal":
        return rule_result + " (AI Confirmed)", True

    if ai_flag:
        return "Anomaly (AI)", True

    return rule_result, False

# -----------------------------------
def get_alert_level(attack):

    if "DDoS" in attack:
        return "HIGH"

    if "SYN" in attack:
        return "CRITICAL"

    if "Port Scan" in attack:
        return "MEDIUM"

    if "Anomaly" in attack:
        return "LOW"

    return "INFO"

# -----------------------------------
def threat_score(flow):

    score = 0
    score += flow["packets"]
    score += flow["syn"] * 2
    score += flow["icmp"] * 2

    return min(score, 100)

# -----------------------------------
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

# -----------------------------------
def block_ip(ip):

    now = time.time()

    if ip in blocked_ips and now - blocked_ips[ip] < COOLDOWN:
        return

    os.system(f"iptables -A INPUT -s {ip} -j DROP")
    blocked_ips[ip] = now

    print(f"🚫 BLOCKED: {ip}")

# -----------------------------------
def cleanup_blocks():

    now = time.time()

    for ip in list(blocked_ips.keys()):
        if now - blocked_ips[ip] > BLOCK_TIME:
            os.system(f"iptables -D INPUT -s {ip} -j DROP")
            del blocked_ips[ip]
            print(f"✅ UNBLOCKED: {ip}")

# -----------------------------------
def process_packet(packet):

    if IP not in packet:
        return

    src = packet[IP].src

    if src not in flows:
        flows[src] = create_flow(packet)

    update_flow(flows[src], packet)
    update_baseline(flows[src])

    if flows[src]["packets"] >= MIN_PACKETS:

        attack, ai_flag = hybrid_decision(flows[src])

        duration = max(flows[src]["last"] - flows[src]["start"], 0.1)
        rate = flows[src]["packets"] / duration

        print(f"[{src}] rate={round(rate,1)} | baseline={round(baseline_rate,1)} -> {attack}")

        if attack != "Normal":

            level = get_alert_level(attack)
            score = threat_score(flows[src])

            print(f"🚨 [{level}] ALERT: {attack}")
            print(f"Threat Score: {score}")

            reasons = explain(flows[src], ai_flag)

            print("Reason:")
            for r in reasons:
                print(f" - {r}")

            log_attack(flows[src], attack)
            block_ip(src)

        cleanup_blocks()
        flows[src] = create_flow(packet)

# -----------------------------------
sniff(prn=process_packet, store=0, iface=["eth0", "lo"])