from fastapi.responses import FileResponse, JSONResponse
from env import Env
env = Env()


def return_url_object(filename: str, obj: str) -> str:
    return (f"http://{env.__getattr__('HOST')}:{env.__getattr__('SERVER_PORT')}/server/"
            f"public/{obj}/{filename}")
