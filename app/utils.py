# app/utils.py

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import zscore

def calculate_smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    smape_values = np.where(denominator == 0, 0, diff / denominator)
    return np.mean(smape_values) * 100

def calculate_mase(training_series, y_true, y_pred):
    training_series = np.array(training_series)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    scale = np.mean(np.abs(training_series[1:] - training_series[:-1]))
    if scale == 0:
        return float('nan')
    return mean_absolute_error(y_true, y_pred) / scale

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터 전처리: 카테고리 필터링 등.
    """
    
    # 미션 생성 카테고리(자동차, 주거통신, 의료건강, 금융, 교육학습, 자녀육아, 반려동물, 경조선물, 이체 제외)
    valid_categories = [
        "FOOD", "CAFE_SNACKS", "TRANSPORTATION", "ONLINE_SHOPPING",
        "ALCOHOL_ENTERTAINMENT", "LIVING", "FASHION_SHOPPING", "BEAUTY_CARE",
        "CULTURE_LEISURE", "TRAVEL_ACCOMMODATION"
    ]
    df = df[df["category"].isin(valid_categories)]
    return df

# 카테고리별 주 단위 집계
def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    주 단위 집계: 주의 첫날 기준 timestamp 생성 후, 카테고리별 소비 금액 합산.
    """
    df["week"] = df["timestamp"].dt.to_period("W").apply(lambda x: x.start_time)
    df = df.drop(columns=["timestamp"]).rename(columns={"week": "timestamp", "amount": "target", "expense_category": "item_id"})
    weekly_df = df.groupby(["item_id", "timestamp"], as_index=False).agg({"target": "sum"})
    
    # 누락된 주 채우기
    start_date = weekly_df["timestamp"].min()
    end_date = weekly_df["timestamp"].max()
    all_weeks = pd.date_range(start=start_date, end=end_date, freq="7D")
    all_categories = weekly_df["item_id"].unique()
    all_combinations = pd.MultiIndex.from_product([all_categories, all_weeks], names=["item_id", "timestamp"])
    all_weeks_df = pd.DataFrame(index=all_combinations).reset_index()
    weekly_df_full = pd.merge(all_weeks_df, weekly_df, on=["item_id", "timestamp"], how="left")
    weekly_df_full["target"] = weekly_df_full["target"].fillna(0)
    return weekly_df_full

def adjust_outliers(weekly_df_full: pd.DataFrame) -> pd.DataFrame:
    """
    이상치 보정: 이동평균, IQR 및 Z-score를 사용한 보정.
    """
    df_budget = weekly_df_full.copy().sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    window_size = 4
    # 이동평균 기반 보정
    df_budget["rolling_mean"] = df_budget.groupby("item_id")["target"].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    df_budget["rolling_std"] = df_budget.groupby("item_id")["target"].transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
    df_budget["is_outlier_ma"] = (df_budget["target"] > df_budget["rolling_mean"] + 1.8 * df_budget["rolling_std"]) | \
                                  (df_budget["target"] < df_budget["rolling_mean"] - 1.8 * df_budget["rolling_std"])
    df_budget["target_original_ma"] = df_budget["target"]
    df_budget.loc[df_budget["is_outlier_ma"], "target"] = df_budget["rolling_mean"]
    
    # IQR 및 Z-score 기반 보정
    Q1 = df_budget["target"].quantile(0.25)
    Q3 = df_budget["target"].quantile(0.75)
    IQR = Q3 - Q1
    df_budget["is_outlier_iqr"] = (df_budget["target"] > (Q3 + 1.5 * IQR)) | (df_budget["target"] < (Q1 - 1.5 * IQR))
    df_budget["z_score"] = np.abs(zscore(df_budget["target"]))
    df_budget["is_outlier_z"] = df_budget["z_score"] > 3
    df_budget["is_outlier"] = df_budget["is_outlier_iqr"] | df_budget["is_outlier_z"]
    df_budget["target_original_iqr"] = df_budget["target"]
    df_budget.loc[df_budget["is_outlier"], "target"] = df_budget["rolling_mean"]
    return df_budget

def run_prophet(final_df: pd.DataFrame) -> (pd.DataFrame):
    """
    각 카테고리별로 Prophet 모델을 학습 및 예측하고 평가 지표 계산.
    """
    results_list = []
    
    # 각 카테고리(item_id)별로 모델 학습 및 예측 수행
    for category in final_df['item_id'].unique():
        group_df = final_df[final_df['item_id'] == category].copy()
        # Prophet용 컬럼명 변환: timestamp -> ds, target -> y
        train_df = group_df.rename(columns={"timestamp": "ds", "target": "y"})
        
        # 로그 변환: 0 이하의 값이 없도록 np.log1p 사용
        train_df["y"] = np.log1p(train_df["y"])
        
        # Prophet 모델 생성 및 학습
        model = Prophet(weekly_seasonality=True, yearly_seasonality=False, changepoint_prior_scale=0.7)
        model.fit(train_df)
        
        # 미래 바로 다음 주 예측을 위해 period를 1로 설정
        future = model.make_future_dataframe(periods=1, freq='W')

        forecast = model.predict(future)
        
        # 예측값 후처리: 음수값 방지 및 로그 변환 복원
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat'] = np.expm1(forecast['yhat'])
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        
        # 학습 데이터 이후의 예측 결과만 추출 (순수 미래 예측)
        last_date = train_df['ds'].max()
        future_forecast = forecast[forecast['ds'] > last_date].copy()

        # 결과 DataFrame 생성: yhat 및 카테고리 정보 추가 (ds 컬럼 제거)
        result = future_forecast[['yhat']].copy()
        result["Category"] = category

        
        # 이동평균 기반 보정: 마지막 8주간의 데이터 범위를 활용하여 보정
        last_date = train_df['ds'].max()
        window_start_date = last_date - pd.Timedelta(weeks=8)
        if window_start_date < train_df['ds'].min():
            window_start_date = train_df['ds'].min()
        
        available_period = last_date - window_start_date
        if available_period < pd.Timedelta(weeks=1):
            result['yhat_adjusted'] = result['yhat']
        else:
            df_window = group_df[(group_df['timestamp'] >= window_start_date) & (group_df['timestamp'] <= last_date)]
            window_min = df_window['target'].min()
            window_max = df_window['target'].max()
            result['window_min'] = window_min
            result['window_max'] = window_max
            result['yhat_adjusted'] = result['yhat'].clip(lower=window_min, upper=window_max)
        
        results_list.append(result)
    
    all_results = pd.concat(results_list, ignore_index=True)
    print(all_results)
    
    return all_results
            