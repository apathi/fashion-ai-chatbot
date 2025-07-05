"""
Fashion AI Chatbot - Flask Application

A production-ready web application demonstrating Retrieval-Augmented Generation (RAG)
for fashion trend recommendations. This app showcases:
- Semantic search using vector embeddings
- Context-aware response generation
- Comparison between basic and RAG-enhanced responses

Author: Technical PM Portfolio Project
"""

import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from src.chatbot import create_chatbot, ConversationManager

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global chatbot instance (initialized on first request)
chatbot = None
conversation_manager = None


def get_chatbot():
    """
    Lazy initialization of chatbot.

    This ensures embeddings are only generated when needed,
    making the app startup faster.
    """
    global chatbot, conversation_manager

    if chatbot is None:
        print("Initializing RAG chatbot...")
        data_path = os.getenv("DATA_PATH", "data/2023_fashion_trends.csv")
        use_cache = os.getenv("CACHE_EMBEDDINGS", "True").lower() == "true"

        chatbot = create_chatbot(data_path, use_cache)
        conversation_manager = ConversationManager(chatbot)
        print("Chatbot initialized successfully!")

    return chatbot, conversation_manager


# Routes

@app.route("/")
def index():
    """Render the main chat interface."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Fashion AI Chatbot",
        "version": "1.0.0"
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint supporting both basic and RAG modes.

    Request JSON:
    {
        "message": "User's question",
        "mode": "rag" or "basic",
        "top_k": 3 (optional, for RAG mode)
    }

    Response JSON:
    {
        "response": "Bot's answer",
        "mode": "rag" or "basic",
        "sources": [...] (only for RAG mode)
    }
    """
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' in request"}), 400

        message = data["message"].strip()
        if not message:
            return jsonify({"error": "Empty message"}), 400

        mode = data.get("mode", "rag").lower()
        top_k = int(data.get("top_k", os.getenv("TOP_K_RESULTS", 3)))

        # Validate mode
        if mode not in ["rag", "basic"]:
            return jsonify({"error": "Mode must be 'rag' or 'basic'"}), 400

        # Get chatbot
        bot, conv_manager = get_chatbot()

        # Get response
        result = conv_manager.send_message(message, mode, top_k)

        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/compare", methods=["POST"])
def compare():
    """
    Compare basic vs RAG responses for the same question.

    Request JSON:
    {
        "message": "User's question",
        "top_k": 3 (optional)
    }

    Response JSON:
    {
        "question": "User's question",
        "basic_response": "...",
        "rag_response": "...",
        "sources": [...]
    }
    """
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' in request"}), 400

        message = data["message"].strip()
        if not message:
            return jsonify({"error": "Empty message"}), 400

        top_k = int(data.get("top_k", os.getenv("TOP_K_RESULTS", 3)))

        # Get chatbot
        bot, _ = get_chatbot()

        # Compare modes
        result = bot.compare_modes(message, top_k)

        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error in compare endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/sample-questions", methods=["GET"])
def sample_questions():
    """
    Get sample questions users can ask.

    Response JSON:
    {
        "questions": [...]
    }
    """
    samples = [
        "Which 2023 fashion trends embrace utilitarian styling?",
        "What are the bold color trends for 2023?",
        "I need polished fashion trends for the office. What should I consider?",
        "What denim trends are popular in 2023?",
        "Tell me about sustainable fashion trends for 2023",
        "What accessories are trending in 2023?",
        "Which 2023 trends work well for casual everyday wear?",
        "What are the key footwear trends for 2023?"
    ]

    return jsonify({"questions": samples})


@app.route("/api/history", methods=["GET"])
def get_history():
    """
    Get conversation history.

    Response JSON:
    {
        "history": [...]
    }
    """
    try:
        _, conv_manager = get_chatbot()
        history = conv_manager.get_history()

        return jsonify({"history": history})

    except Exception as e:
        app.logger.error(f"Error getting history: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    """
    Clear conversation history.

    Response JSON:
    {
        "message": "History cleared"
    }
    """
    try:
        _, conv_manager = get_chatbot()
        conv_manager.clear_history()

        return jsonify({"message": "History cleared"})

    except Exception as e:
        app.logger.error(f"Error clearing history: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# Error handlers

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)

    # Run app
    debug_mode = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    port = int(os.getenv("PORT", 5000))

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║         Fashion AI Chatbot - RAG Demo                     ║
    ║         Retrieval-Augmented Generation Showcase           ║
    ╚═══════════════════════════════════════════════════════════╝

    🚀 Starting server on http://localhost:{port}
    📊 Mode: {'Development' if debug_mode else 'Production'}
    🤖 RAG System: Initializing on first request...

    """)

    app.run(debug=debug_mode, port=port, host="0.0.0.0")
