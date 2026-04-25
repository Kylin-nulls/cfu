from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from streamlit.web import bootstrap


APP_NAME = "ColonyCounter"
DEFAULT_PORT = 8502


def resource_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def user_log_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / APP_NAME
    return Path.home() / ".colony_counter"


def configure_logging() -> Path:
    log_dir = user_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        log_dir = Path(tempfile.gettempdir()) / APP_NAME
        log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "launcher.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return log_file


def find_free_port(preferred: int = DEFAULT_PORT) -> int:
    for port in [preferred, 8501, 8503, 8504, 8505, 0]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return int(sock.getsockname()[1])
    raise RuntimeError("No free local port found")


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if 200 <= response.status < 500:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.35)
    return False


def open_browser_when_ready(url: str) -> None:
    if wait_for_server(url):
        webbrowser.open(url)
    else:
        logging.warning("Timed out waiting for Streamlit server: %s", url)


def show_error(message: str) -> None:
    logging.exception(message)
    if os.name != "nt":
        print(message, file=sys.stderr)
        return

    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10)
    except Exception:
        print(message, file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the colony counter desktop app.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CFU_PORT", DEFAULT_PORT)))
    parser.add_argument("--no-browser", action="store_true", default=os.environ.get("CFU_OPEN_BROWSER") == "0")
    parser.add_argument("--test", action="store_true", help="Validate packaged resources and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_file = configure_logging()
    base_dir = resource_dir()
    app_path = base_dir / "app.py"

    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    if not app_path.exists():
        raise FileNotFoundError(f"Cannot find Streamlit app: {app_path}")

    if args.test:
        print(f"OK: {app_path}")
        print(f"Log: {log_file}")
        return 0

    port = find_free_port(args.port)
    url = f"http://localhost:{port}"
    logging.info("Starting %s from %s on %s", APP_NAME, app_path, url)

    if not args.no_browser:
        threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()

    flag_options = {
        "server.port": port,
        "server.headless": True,
        "server.fileWatcherType": "none",
        "server.runOnSave": False,
        "browser.gatherUsageStats": False,
        "global.developmentMode": False,
    }
    bootstrap.run(str(app_path), False, [], flag_options)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        show_error(f"启动失败：{exc}\n日志文件：{user_log_dir() / 'launcher.log'}")
        raise SystemExit(1)
