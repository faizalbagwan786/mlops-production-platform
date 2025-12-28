from pydantic import BaseModel

class ChurnRequest(BaseModel):
    age: int
    monthly_charges: float
    tenure: int
    contract_type: str
