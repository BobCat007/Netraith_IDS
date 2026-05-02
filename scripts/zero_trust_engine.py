import time
from collections import defaultdict


class ZeroTrustEngine:

    def __init__(self):

        # Trust score table (default 0.5)
        self.trust_scores = defaultdict(lambda: 0.5)

        # Track last seen time
        self.last_seen = {}

    # -------------------------------
    # Update Trust Score
    # -------------------------------

    def update_behavior(self, ip, anomaly_score):

        score = self.trust_scores[ip]

        # decrease trust for anomalies
        if anomaly_score > 0.7:
            score -= 0.3

        elif anomaly_score > 0.4:
            score -= 0.15

        else:
            score += 0.05

        # clamp score between 0 and 1
        score = max(0.0, min(1.0, score))

        self.trust_scores[ip] = score
        self.last_seen[ip] = time.time()

        return score

    # -------------------------------
    # Zero Trust Decision
    # -------------------------------

    def decision(self, ip):

        score = self.trust_scores[ip]

        if score < 0.2:
            return "BLOCK"

        elif score < 0.4:
            return "THROTTLE"

        elif score < 0.6:
            return "MONITOR"

        else:
            return "ALLOW"

    # -------------------------------
    # Print Trust Table
    # -------------------------------

    def print_table(self):

        print("\nZero Trust Table")

        for ip, score in self.trust_scores.items():

            decision = self.decision(ip)

            print(f"{ip} | Trust Score: {score:.2f} | {decision}")
