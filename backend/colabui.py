# Step 1: Setup & Imports
!pip install flask flask-cors chainlit pyngrok --quiet

import os
import threading
import subprocess
import time
from pyngrok import ngrok

# Set environment for Flask CLI
os.environ["FLASK_APP"] = "app.py"
os.environ["FLASK_ENV"] = "development"

# 🔥 Thread: Run Flask via CLI
def run_flask():
    # Runs app.py via Flask CLI
    subprocess.run(["flask", "run", "--host=0.0.0.0", "--port=5000"])

# 🔥 Thread: Run Chainlit
def run_chainlit():
    subprocess.run(["chainlit", "run", "ui.py", "--port", "8000"])

# 🚀 Start both threads
flask_thread = threading.Thread(target=run_flask)
chainlit_thread = threading.Thread(target=run_chainlit)

flask_thread.start()
time.sleep(5)  # wait for Flask to start
chainlit_thread.start()

# 🌐 Open ngrok tunnels
flask_url = ngrok.connect(5000)
chainlit_url = ngrok.connect(8000)

print(f"🔗 Flask API: {flask_url}")
print(f"💬 Chainlit UI: {chainlit_url}")
