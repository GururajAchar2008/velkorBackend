"""
app.py

Main entry point for the Velkor AI / Guru JI AI Backend.
Registers blueprints and initializes the Flask application.
"""

from flask import Flask, jsonify
from flask_cors import CORS

from config import Config
from routers.chat import chat_bp
from routers.health import health_bp
from routers.image import image_bp
from routers.models import models_bp
from utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_ORIGINS = [
    "https://gururajachar2008.github.io",
]


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Enable CORS for the frontend. Only the deployed GitHub Pages origin
    # is allowed to call the API.
    CORS(
        app,
        origins=ALLOWED_ORIGINS,
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        supports_credentials=False,
    )

    # Register Blueprints
    app.register_blueprint(chat_bp, url_prefix="/api")
    app.register_blueprint(health_bp)
    app.register_blueprint(models_bp, url_prefix="/api")
    app.register_blueprint(image_bp, url_prefix="/api")

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
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
