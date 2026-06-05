import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

FASTAPI_BASE_URL = os.environ.get("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/files", methods=["GET"])
def files():
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/files", timeout=10)
        if response.status_code == 200:
            return jsonify(response.json())
        return jsonify({"files": []})
    except Exception as e:
        return jsonify({"error": str(e), "files": []}), 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data or not data["question"].strip():
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/ask", json=data, timeout=60)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            try:
                err_detail = response.json().get("detail", response.text)
            except Exception:
                err_detail = response.text
            return jsonify({"error": f"Backend Error: {err_detail}"}), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "The RAG backend server is currently offline. Please ensure the FastAPI server is running on port 8000."
        }), 503
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
