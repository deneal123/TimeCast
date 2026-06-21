import os


def return_url_object(filename: str, obj: str) -> str:
    host = os.getenv("HOST", "localhost")
    port = os.getenv("SERVER_PORT", "8000")
    return f"http://{host}:{port}/server/public/{obj}/{filename}"
