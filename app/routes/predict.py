# app/routes/predict.py

from fastapi import APIRouter, HTTPException
import pandas as pd
from app.models import DataRequest
from app.utils import (
    preprocess_data,
    aggregate_weekly,
    adjust_outliers,
    run_prophet
)

router = APIRouter()

# 12주 데이터 받아와야 함
@router.post("/predict")
def predict_endpoint(data_request: DataRequest):
    try:
        # 1. 입력 데이터를 DataFrame으로 변환
        df = pd.DataFrame([record.dict() for record in data_request.data])
        
        # 2. 전처리
        df = preprocess_data(df)
        
        # 3. 주 단위 집계
        weekly_df_full = aggregate_weekly(df)
        
        # 4. 이상치 보정
        adjusted_df = adjust_outliers(weekly_df_full)
        final_df = adjusted_df[["item_id", "timestamp", "target"]].copy()
        final_df["timestamp"] = pd.to_datetime(final_df["timestamp"])
        
        # 5. Prophet 모델 학습, 예측 및 평가
        results_df = run_prophet(final_df)
        
        predict_results = results_df[['Category', 'yhat_adjusted']].to_dict(orient="records")

        return {"predict_results": predict_results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
