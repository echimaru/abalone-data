import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from catboost import CatBoostRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(page_title="전복 나이 예측 대시보드", layout="wide")
st.title("전복 나이(Rings) 예측 시스템")
st.caption("CatBoost 모델 기반 | Train/Test split 80:20 | R² ≈ 0.61")

# ─────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    column_names = [
        'Sex', 'Length', 'Diameter', 'Height',
        'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'
    ]
    df = pd.read_csv('abalone.data.csv', header=None, names=column_names)
    return df

try:
    df_raw = load_data()
except FileNotFoundError:
    st.error("abalone.data.csv 파일을 찾을 수 없습니다.")
    st.stop()

# ─────────────────────────────────────────
# 2. 전처리 함수
# ─────────────────────────────────────────
def remove_outliers(df):
    df = df[(df['Height'] > 0) & (df['Height'] < 0.4)]
    df = df[df['Whole weight'] >= df['Shucked weight'] + df['Viscera weight'] + df['Shell weight']]
    df = df[df['Shucked weight'] >= df['Viscera weight']]
    df = df[df['Whole weight'] >= df['Shucked weight'] + df['Shell weight']]
    return df

def feature_engineering(df):
    df = df.copy()
    df["Shell Water"] = df["Whole weight"] - (df["Shucked weight"] + df["Shell weight"])
    df["density"] = df["Shucked weight"] / (df["Length"] * df["Diameter"] * df["Height"] + 1e-9)
    df["Shell ratio"] = df["Shell weight"] / (df["Whole weight"] + 1e-9)
    df["Meat_Ratio"] = df["Shucked weight"] / (df["Whole weight"] + 1e-9)
    return df

# ─────────────────────────────────────────
# 3. 모델 학습 / 로드
# ─────────────────────────────────────────
MODEL_PATH = 'abalone_catboost_v2.pkl'
COLUMNS_PATH = 'model_columns.pkl'

@st.cache_resource
def get_model_and_meta():
    # 전처리
    df = remove_outliers(df_raw.copy())
    df = feature_engineering(df)
    df_model = pd.get_dummies(df, columns=['Sex'], drop_first=False, dtype=int)
    X = df_model.drop('Rings', axis=1)
    y = df_model['Rings']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(COLUMNS_PATH)
    else:
        model = CatBoostRegressor(
            iterations=1000, learning_rate=0.03, depth=6,
            subsample=0.8, colsample_bylevel=0.8,
            random_state=42, early_stopping_rounds=50, verbose=0
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        model_columns = X.columns.tolist()
        joblib.dump(model, MODEL_PATH)
        joblib.dump(model_columns, COLUMNS_PATH)

    # RMSE 계산 (신뢰구간용)
    y_pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return model, model_columns, X, df, rmse

with st.spinner('모델 로드 중...'):
    model, model_columns, X_full, df_clean, rmse = get_model_and_meta()

# ─────────────────────────────────────────
# 4. 사이드바 입력
# ─────────────────────────────────────────
st.sidebar.header("특성 입력 (Features)")

def user_input():
    shell_weight   = st.sidebar.number_input("Shell weight (껍질 무게)",   min_value=0.0, value=float(df_raw['Shell weight'].mean()),   format="%.4f")
    shucked_weight = st.sidebar.number_input("Shucked weight (살 무게)",   min_value=0.0, value=float(df_raw['Shucked weight'].mean()), format="%.4f")
    whole_weight   = st.sidebar.number_input("Whole weight (전체 무게)",   min_value=0.0, value=float(df_raw['Whole weight'].mean()),   format="%.4f")
    viscera_weight = st.sidebar.number_input("Viscera weight (내장 무게)", min_value=0.0, value=float(df_raw['Viscera weight'].mean()),format="%.4f")
    diameter       = st.sidebar.number_input("Diameter (지름)",            min_value=0.0, value=float(df_raw['Diameter'].mean()),       format="%.4f")
    length         = st.sidebar.number_input("Length (길이)",              min_value=0.0, value=float(df_raw['Length'].mean()),         format="%.4f")
    height         = st.sidebar.number_input("Height (높이)",              min_value=0.0, value=float(df_raw['Height'].mean()),         format="%.4f")
    sex            = st.sidebar.selectbox("Sex (성별)", ('M', 'F', 'I'))
    return pd.DataFrame({
        'Sex': sex, 'Length': length, 'Diameter': diameter, 'Height': height,
        'Whole weight': whole_weight, 'Shucked weight': shucked_weight,
        'Viscera weight': viscera_weight, 'Shell weight': shell_weight
    }, index=[0])

input_df = user_input()

# ─────────────────────────────────────────
# 5. 입력값 전처리 → 예측
# ─────────────────────────────────────────
input_processed = feature_engineering(input_df)
input_processed = pd.get_dummies(input_processed, columns=['Sex'], drop_first=False, dtype=int)
for col in model_columns:
    if col not in input_processed.columns:
        input_processed[col] = 0
input_processed = input_processed[model_columns]

prediction = float(np.maximum(model.predict(input_processed), 0)[0])
predicted_age = prediction + 1.5

# ─────────────────────────────────────────
# 6. KNN 유사 전복 분석
# ─────────────────────────────────────────
@st.cache_resource
def get_knn(_X_full, _model_columns):
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(_X_full[_model_columns])
    return knn

knn_model = get_knn(X_full, model_columns)
_, indices = knn_model.kneighbors(input_processed)
similar_samples = df_clean.iloc[indices[0]]
similar_rings_mean = similar_samples['Rings'].mean()

# ─────────────────────────────────────────
# 7. 결과 출력
# ─────────────────────────────────────────
st.divider()

st.subheader("예측 결과")

# 핵심 예측값
m1, m2 = st.columns(2)
m1.metric("예측 나이테 (Rings)", f"{prediction:.2f}", delta=f"± {rmse:.2f} (RMSE)", delta_color="off")
m2.metric("예상 실제 나이", f"약 {predicted_age:.1f} 세", delta=f"{predicted_age - rmse:.1f} ~ {predicted_age + rmse:.1f} 세", delta_color="off")

# 유사 전복 근거
st.info(f"가장 유사한 전복 5마리의 평균 나이테: **{similar_rings_mean:.1f}개** (KNN 기반)")

with st.expander("🧐 가장 유사한 전복 5마리 데이터 보기"):
    st.dataframe(similar_samples)

# 8. 추가 분석 리포트 (백분위수)
# 전체 데이터 대비 현재 입력값의 위치 계산
pct_weight = (df_raw['Whole weight'] < input_df['Whole weight'].iloc[0]).mean() * 100
pct_length = (df_raw['Length'] < input_df['Length'].iloc[0]).mean() * 100

st.markdown(f"""
    > 📊 **분석 리포트**  
    > 입력하신 전복은 전체 데이터 집단 내에서:  
    > - **무게(Whole weight)** 기준: 상위 **{100-pct_weight:.1f}%** (하위 {pct_weight:.1f}%)  
    > - **길이(Length)** 기준: 상위 **{100-pct_length:.1f}%** (하위 {pct_length:.1f}%)  
    > 에 해당합니다.
""")

st.divider()

st.subheader("데이터 분포 분석 (2D)")

# 2D 산점도 변수 선택 (숫자형 변수만 필터링)
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

c1, c2 = st.columns(2)
with c1:
    x_axis = st.selectbox("X축 변수", numeric_cols, index=numeric_cols.index('Shell weight') if 'Shell weight' in numeric_cols else 0)
with c2:
    y_axis = st.selectbox("Y축 변수", numeric_cols, index=numeric_cols.index('Rings') if 'Rings' in numeric_cols else 0)

fig_2d = px.scatter(
    df_clean, x=x_axis, y=y_axis,
    opacity=0.3, color='Rings', color_continuous_scale='Viridis',
    title=f"전체 분포 ({x_axis} vs {y_axis})"
)

# 내 전복 위치 (Rings 선택 시 예측값 사용, 그 외는 입력값 사용)
my_x = prediction if x_axis == 'Rings' else input_processed[x_axis].values[0]
my_y = prediction if y_axis == 'Rings' else input_processed[y_axis].values[0]

fig_2d.add_scatter(
    x=[my_x], y=[my_y],
    mode='markers',
    marker=dict(size=25, color='#ff2b2b', symbol='star', line=dict(width=2, color='black')),
    name='내 전복'
)
fig_2d.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=350, showlegend=True)
st.plotly_chart(fig_2d, use_container_width=True)
