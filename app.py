"""
app.py

Main entry point for the Velkor AI Backend.
Registers HTTP blueprints and Flask-SocketIO for streaming chat / image gen.
"""

from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO

from config import Config
from routers.chat import chat_bp
from routers.health import health_bp
from routers.image import image_bp
from routers.models import models_bp
from routers.upload import upload_bp
from routers.memory import memory_bp
from routers.socket_events import register_socket_events
from utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_ORIGINS = [
    "https://gururajachar2008.github.io",
    "https://gururajachar2008.github.io/Velkor",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

# Threading mode is portable across Python versions and works with
# gunicorn gthread workers. Eventlet is avoided due to 3.12+ breakage.
socketio = SocketIO(
    cors_allowed_origins=ALLOWED_ORIGINS,
    async_mode="threading",
    logger=False,
    engineio_logger=False,
    ping_timeout=120,
    ping_interval=25,
)


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

    CORS(
        app,
        origins=ALLOWED_ORIGINS,
        methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        supports_credentials=False,
    )

    app.register_blueprint(chat_bp, url_prefix="/api")
    app.register_blueprint(health_bp)
    app.register_blueprint(models_bp, url_prefix="/api")
    app.register_blueprint(image_bp, url_prefix="/api")
    app.register_blueprint(upload_bp, url_prefix="/api")
    app.register_blueprint(memory_bp, url_prefix="/api")

    socketio.init_app(app)
    register_socket_events(socketio)

    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "status": "online",
            "service": "Velkor AI Backend",
            "version": "3.0",
            "socketio": True,
        }), 200

    return app


app = create_app()

if __name__ == "__main__":
    socketio.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        allow_unsafe_werkzeug=True,
    )
