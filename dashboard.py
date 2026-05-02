import eventlet
eventlet.monkey_patch()   # MUST be first

from flask import Flask, render_template
from flask_socketio import SocketIO
import pandas as pd

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

LOG_FILE = "ids_logs.csv"

# -------------------------------
def get_data():
    try:
        df = pd.read_csv(LOG_FILE)

        return {
            "logs": df.tail(10).to_dict(orient="records"),
            "total": int(len(df)),
            "attackers": int(df["Source IP"].nunique())
        }
    except Exception as e:
        print("ERROR:", e)
        return {"logs": [], "total": 0, "attackers": 0}

# -------------------------------
def background_task():
    print("Background task started")

    while True:
        data = get_data()
        socketio.emit("update", data)
        socketio.sleep(2)

# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
@socketio.on("connect")
def handle_connect():
    print("CLIENT CONNECTED")
    socketio.start_background_task(background_task)

# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting REAL SocketIO server...")
    socketio.run(app, host="0.0.0.0", port=5000)