def generate_report(attack):

    reports = {

        "PortScan": "Port scanning detected. Attacker probing open ports.",

        "DDoS": "Distributed denial of service attack detected.",

        "Bot": "Botnet activity detected.",

        "Infiltration": "Unauthorized system access attempt."

    }

    return reports.get(attack, "Unknown threat detected")


print(generate_report("PortScan"))
