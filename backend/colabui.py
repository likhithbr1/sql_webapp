import threading
import subprocess
import time
from pyngrok import ngrok

# Start Flask in a thread
def run_flask():
    subprocess.run(["python3", "app.py"])  # Your existing Flask backend

# Start Chainlit in a thread
def run_chainlit():
    subprocess.run(["chainlit", "run", "ui.py", "--port", "8000"])  # Your Chainlit app file

# Start threads
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

time.sleep(5)  # Let Flask start first

chainlit_thread = threading.Thread(target=run_chainlit)
chainlit_thread.start()

# Start ngrok tunnels
flask_url = ngrok.connect(5000)
chainlit_url = ngrok.connect(8000)

print(f"\n🔗 Flask API available at: {flask_url}")
print(f"💬 Chainlit UI available at: {chainlit_url}")
