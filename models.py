from pydantic import BaseModel
from typing import Dict, Any

class TestModel(BaseModel):
    x: int

t= TestModel(x=True)
print(t)