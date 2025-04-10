from flask import Flask, request, jsonify
from flask_cors import CORS
from core_logic import process_question  # <- import your main function

# -------------------------------------
# Initialize Flask app
# -------------------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# -------------------------------------
# Health Check Endpoint
# -------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

# -------------------------------------
# Main Chatbot Endpoint
# -------------------------------------
@app.route("/api/query", methods=["POST"])
def handle_query():
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        question = data["question"].strip()
        if not question:
            return jsonify({"error": "Empty question"}), 400

        # Call your core logic
        result = process_question(question)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------
# Run the app (local dev)
# -------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
