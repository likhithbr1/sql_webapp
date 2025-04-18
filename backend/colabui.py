import threading
import subprocess
import time
from pyngrok import ngrok

# ✅ Directly run Flask from app.py's app
def run_flask_direct():
    import app
    app.app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# ✅ Run chainlit via subprocess
def run_chainlit():
    subprocess.run(["chainlit", "run", "ui.py", "--port", "8000"])

# 🔥 Start Flask thread
flask_thread = threading.Thread(target=run_flask_direct)
flask_thread.start()

# ⏱ Wait for Flask to spin up
time.sleep(5)

# 🔥 Start Chainlit thread
chainlit_thread = threading.Thread(target=run_chainlit)
chainlit_thread.start()

# 🌐 Open tunnels
flask_url = ngrok.connect(5000)
chainlit_url = ngrok.connect(8000)

print(f"\n🔗 Flask API: {flask_url}")
print(f"💬 Chainlit UI: {chainlit_url}")
