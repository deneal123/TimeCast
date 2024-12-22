from pydantic import (BaseModel, Field, StrictStr, Json, condecimal,
                      StrictInt, PrivateAttr, SecretBytes, StrictBytes, StrictBool, root_validator,
                      SecretStr)
from enum import Enum
from typing import Optional, List, ClassVar
from datetime import datetime
import os
from pathlib import Path
from env import Env
from cryptography.fernet import Fernet
from datetime import datetime
from uuid import uuid4

env = Env()




class Images(BaseModel):
    """
    Model of images
    """
    ID: Optional[int] = Field(None,
                              alias="id")
    Url: StrictStr = Field(...,
                           alias="url",
                           examples=["https://example.com"],
                           description="URL of images")