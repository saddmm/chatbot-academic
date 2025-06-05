from flask import Flask, jsonify, request

from app.chain import run_chatbot


app = Flask(__name__)


@app.route("/")
def home():
    return "Hello, World! This is the API for Prodi Assistant."


@app.route("chat", methods=["POST"])
async def chat():
    # Placeholder for chat functionality
    try:
        message = request.json.get("message")
        if not message:
            return {"error": "Message is required"}, 400

        response = await run_chatbot(message)
        return jsonify({response: response}), 200
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)  # Set debug=True for development purposes
