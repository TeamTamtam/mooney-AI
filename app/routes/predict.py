# app/routes/predict.py

from fastapi import APIRouter, HTTPException
import pandas as pd
from app.models import DataRequest
from app.utils import (
    preprocess_data,
    aggregate_weekly,
    adjust_outliers,
    run_prophet,
    select_good_categories
)

router = APIRouter()

# 12주 데이터 받아와야 함
@router.post("/predict")
def predict_endpoint(data_request: DataRequest):
    import time
    total_start = time.time()

    try:
        # 1. 입력 데이터를 DataFrame으로 변환
        # 백엔드에서 이미 전처리 및 주 단위 집계(12주)된 데이터가 전달
        print("Received JSON:", data_request)  # ✅ JSON 확인
        df = pd.DataFrame([record.dict() for record in data_request.data])
        print("Converted DataFrame:", df)  # ✅ DataFrame 변환 확인
        
        # # 2. 전처리 => 백엔드에서 처리 
        # df = preprocess_data(df)
        
        # # 3. 주 단위 집계
        # weekly_df_full = aggregate_weekly(df)
        
         # ✅ 컬럼명 변경: expense_category → item_id
        df = df.rename(columns={"expense_category": "item_id"})
        df = df.rename(columns={"amount": "target"})


        print("✅ Renamed DataFrame:", df)  # 컬럼명 변경 확인
        
        # 4. 예측이 유의미한 카테고리 뽑기
        selected_df = select_good_categories(df)
        
        # 4. 이상치 보정
        adjusted_df = adjust_outliers(selected_df)
        final_df = adjusted_df[["item_id", "timestamp", "target"]].copy()
        final_df["timestamp"] = pd.to_datetime(final_df["timestamp"])
        
        # 5. Prophet 모델 학습, 예측 및 평가
        results_df = run_prophet(final_df)
        
        predict_results = results_df[['Category', 'yhat_adjusted']].to_dict(orient="records")

        total_end = time.time()
        print(f"/predict 요청 처리 총 시간: {total_end - total_start:.3f}초")
        return {"predict_results": predict_results}
    
    except Exception as e:
        print("Error:", str(e))  # ✅ 오류 메시지 출력
        raise HTTPException(status_code=500, detail=str(e))
