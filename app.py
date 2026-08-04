"""
app.py

Main entry point for the Velkor AI / Guru JI AI Backend.
Registers blueprints and initializes the Flask application.
"""

from flask import Flask, jsonify
from flask_cors import CORS
from config import Config
from routes.chat import chat_bp
from utils.logger import get_logger

logger = get_logger(__name__)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Enable CORS for frontend integration
    CORS(app)

    # Register Blueprints
    app.register_blueprint(chat_bp, url_prefix="/api")

    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "status": "online",
            "service": "Velkor AI / Guru JI AI Backend",
            "version": "2.0"
        }), 200

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)