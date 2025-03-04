# app/models.py

from pydantic import BaseModel
from typing import List

class Record(BaseModel): # 지출 가져오기 - 백엔드에서 가져오기 
    timestamp: str         # 예: "2024-08-12"
    # payee: str #결제처 - 미션 만들 때 해당 카테고리에 대해서만 가져오면 됨
    amount: int
    expense_category: str       # 예: "식비", "교통" 등

class DataRequest(BaseModel): # 최근 12주의 데이터 가져오기 
    data: List[Record]
