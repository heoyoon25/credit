import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# 한글 폰트 설정 (깨짐 방지용, 환경에 따라 수정 필요)
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 세션 상태 초기화
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'df_preprocessed' not in st.session_state:
    st.session_state['df_preprocessed'] = None
if 'models' not in st.session_state:
    st.session_state['models'] = {}
if 'split_data' not in st.session_state:
    st.session_state['split_data'] = {}

# 사이드바: 페이지 네비게이션
st.sidebar.title("네비게이션")
page = st.sidebar.radio("페이지 선택", 
                        ["1. 신용평가모형 (메인)", 
                         "2. 데이터 탐색", 
                         "3. 데이터 전처리 및 분할", 
                         "4. 연구 모형", 
                         "5. 연구 결과"])

# ----------------------------------------------------------------------
# 페이지 1: 메인 페이지 (데이터 업로드)
# ----------------------------------------------------------------------
if page == "1. 신용평가모형 (메인)":
    st.title('신용평가모형')
    st.write("이탈 고객 예측을 위한 신용평가 모델 구축 파이프라인입니다.")
    st.write("사용하실 데이터 파일(`Accepted_data (1).csv`)을 아래에 업로드해 주세요.")
    
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df
            st.session_state['df_preprocessed'] = df.copy() # 전처리용 데이터 백업
            st.success("데이터가 성공적으로 업로드되었습니다! 다음 페이지로 이동하세요.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"데이터를 읽는 중 오류가 발생했습니다: {e}")

# ----------------------------------------------------------------------
# 페이지 2: 데이터 탐색 (EDA)
# ----------------------------------------------------------------------
elif page == "2. 데이터 탐색":
    st.header('데이터 탐색')
    df = st.session_state['df']
    
    if df is not None:
        st.subheader("데이터 기본 정보")
        col1, col2 = st.columns(2)
        col1.metric("행의 갯수 (Rows)", df.shape[0])
        col2.metric("열의 갯수 (Columns)", df.shape[1])
        
        st.write("**변수 목록 및 타입**")
        df_types = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index()
        df_types.rename(columns={'index': 'Variable'}, inplace=True)
        st.dataframe(df_types, use_container_width=True)
        
        st.divider()
        st.subheader("시각화 도구")
        
        columns = df.columns.tolist()
        x_var = st.selectbox("X축 변수 선택", options=columns)
        y_var = st.selectbox("Y축 변수 선택 (선택 사항)", options=['선택 안함'] + columns)
        
        chart_type = st.selectbox("그래프 종류 선택", 
                                  ['Histogram', 'Box plot', 'Scatter plot', 'Bar chart', 'Line chart'])
        
        if st.button("그래프 그리기"):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            try:
                if chart_type == 'Histogram':
                    sns.histplot(data=df, x=x_var, kde=True, ax=ax)
                elif chart_type == 'Box plot':
                    if y_var != '선택 안함':
                        sns.boxplot(data=df, x=x_var, y=y_var, ax=ax)
                    else:
                        sns.boxplot(data=df, y=x_var, ax=ax)
                elif chart_type == 'Scatter plot':
                    if y_var != '선택 안함':
                        sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax)
                    else:
                        st.warning("Scatter plot은 Y축 변수가 필요합니다.")
                elif chart_type == 'Bar chart':
                    if y_var != '선택 안함':
                        sns.barplot(data=df, x=x_var, y=y_var, ax=ax)
                    else:
                        df[x_var].value_counts().plot(kind='bar', ax=ax)
                elif chart_type == 'Line chart':
                    if y_var != '선택 안함':
                        sns.lineplot(data=df, x=x_var, y=y_var, ax=ax)
                    else:
                        st.warning("Line chart는 Y축 변수가 필요하거나, X축의 추세가 명확해야 합니다.")
                
                st.pyplot(fig)
            except Exception as e:
                st.error(f"그래프를 그리는 중 오류가 발생했습니다. 변수의 데이터 타입을 확인해주세요. (에러: {e})")
    else:
        st.warning("1페이지에서 데이터를 먼저 업로드해주세요.")

# ----------------------------------------------------------------------
# 페이지 3: 데이터 전처리, Feature Selection, Data Partitioning
# ----------------------------------------------------------------------
elif page == "3. 데이터 전처리 및 분할":
    st.header('데이터 전처리, Feature Selection, Data Partitioning')
    
    if st.session_state['df_preprocessed'] is not None:
        df_prep = st.session_state['df_preprocessed']
        
        st.subheader("1. 데이터 전처리")
        col1, col2, col3 = st.columns(3)
        
        if col1.button("결측치 처리 (제거)"):
            df_prep = df_prep.dropna()
            st.session_state['df_preprocessed'] = df_prep
            st.success(f"결측치가 제거되었습니다. 현재 행 수: {df_prep.shape[0]}")
            
        if col2.button("이상치 처리 (수치형만)"):
            num_cols = df_prep.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                Q1 = df_prep[col].quantile(0.25)
                Q3 = df_prep[col].quantile(0.75)
                IQR = Q3 - Q1
                df_prep = df_prep[(df_prep[col] >= Q1 - 1.5 * IQR) & (df_prep[col] <= Q3 + 1.5 * IQR)]
            st.session_state['df_preprocessed'] = df_prep
            st.success(f"이상치가 처리되었습니다. 현재 행 수: {df_prep.shape[0]}")
            
        if col3.button("원핫인코딩 (범주형)"):
            df_prep = pd.get_dummies(df_prep, drop_first=True)
            st.session_state['df_preprocessed'] = df_prep
            st.success("범주형 변수에 원핫인코딩이 적용되었습니다.")
            
        st.write("현재 데이터 미리보기:")
        st.dataframe(df_prep.head())
        
        st.divider()
        st.subheader("2. Feature Selection")
        columns = df_prep.columns.tolist()
        
        target_var = st.selectbox("종속변수(Y)를 선택하세요", options=columns)
        
        feature_options = [c for c in columns if c != target_var]
        selected_features = st.multiselect("독립변수(X)를 선택하세요", options=feature_options, default=feature_options)
        
        st.divider()
        st.subheader("3. Data Partitioning")
        ratio = st.radio("Train : Test 비율 선택", options=["7:3", "8:2"])
        test_size = 0.3 if ratio == "7:3" else 0.2
        
        if st.button("데이터 분할 (Train/Test Split)"):
            if not selected_features:
                st.error("최소 하나 이상의 독립변수(X)를 선택해야 합니다.")
            else:
                X = df_prep[selected_features]
                y = df_prep[target_var]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                st.session_state['split_data'] = {
                    'X_train': X_train, 'X_test': X_test,
                    'y_train': y_train, 'y_test': y_test
                }
                st.success(f"데이터가 {ratio} 비율로 성공적으로 분할되었습니다!")
                st.write(f"- Train data shape: {X_train.shape}")
                st.write(f"- Test data shape: {X_test.shape}")

    else:
        st.warning("1페이지에서 데이터를 먼저 업로드해주세요.")

# ----------------------------------------------------------------------
# 페이지 4: 연구 모형
# ----------------------------------------------------------------------
elif page == "4. 연구 모형":
    st.header('연구 모형')
    
    if 'X_train' in st.session_state.get('split_data', {}):
        st.write("분석에 사용할 모형을 선택하고 학습을 진행하세요.")
        
        model_choice = st.multiselect("연구 모형 선택", 
                                      options=["Logistic Regression", "Decision Tree"],
                                      default=["Logistic Regression", "Decision Tree"])
        
        if st.button("모형 학습하기"):
            X_train = st.session_state['split_data']['X_train']
            y_train = st.session_state['split_data']['y_train']
            
            st.session_state['models'] = {} # 기존 모델 초기화
            
            # --- 수정한 예외 처리(try-except) 블록 ---
            try:
                with st.spinner('학습 중입니다...'):
                    if "Logistic Regression" in model_choice:
                        lr = LogisticRegression(max_iter=1000)
                        lr.fit(X_train, y_train)
                        st.session_state['models']["Logistic Regression"] = lr
                        
                    if "Decision Tree" in model_choice:
                        dt = DecisionTreeClassifier(random_state=42)
                        dt.fit(X_train, y_train)
                        st.session_state['models']["Decision Tree"] = dt
                        
                st.success("선택된 모형의 학습이 완료되었습니다! 5페이지에서 결과를 확인하세요.")
                
            except ValueError as ve:
                st.error(f"데이터 값 에러가 발생했습니다: {ve}")
                st.info("💡 팁: 3페이지로 돌아가서 '결측치 제거'와 '원핫인코딩'을 실행했는지, 혹은 종속변수(Y)가 범주형(분류 목적)이 맞는지 확인해 주세요.")
            except Exception as e:
                st.error(f"알 수 없는 에러가 발생했습니다: {e}")
            # ---------------------------------------
    else:
        st.warning("3페이지에서 데이터 전처리 및 분할(Data Partitioning)을 먼저 완료해주세요.")

# ----------------------------------------------------------------------
# 페이지 5: 연구 결과
# ----------------------------------------------------------------------
elif page == "5. 연구 결과":
    st.header('연구 결과')
    
    models = st.session_state.get('models', {})
    split_data = st.session_state.get('split_data', {})
    
    if models and 'X_test' in split_data:
        X_test = split_data['X_test']
        y_test = split_data['y_test']
        
        st.subheader("모형별 성능 평가 지표")
        
        results = []
        roc_data = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            try:
                prec = precision_score(y_test, y_pred, average='binary', pos_label=model.classes_[1])
                rec = recall_score(y_test, y_pred, average='binary', pos_label=model.classes_[1])
                f1 = f1_score(y_test, y_pred, average='binary', pos_label=model.classes_[1])
            except ValueError:
                prec = precision_score(y_test, y_pred, average='macro')
                rec = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
                
            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            })
            
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                y_test_numeric = (y_test == model.classes_[1]).astype(int) 
                fpr, tpr, _ = roc_curve(y_test_numeric, y_prob)
                roc_auc = auc(fpr, tpr)
                roc_data[name] = (fpr, tpr, roc_auc)
                
        results_df = pd.DataFrame(results).set_index("Model")
        st.table(results_df.style.format("{:.4f}"))
        
        st.divider()
        st.subheader("ROC Curve 비교")
        
        if roc_data:
            fig, ax = plt.subplots(figsize=(8, 6))
            for name, (fpr, tpr, roc_auc) in roc_data.items():
                ax.plot(fpr, tpr, lw=2, label=f'{name} (area = {roc_auc:.2f})')
            
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC)')
            ax.legend(loc="lower right")
            
            st.pyplot(fig)
        else:
            st.write("ROC Curve를 그리기 위한 확률(predict_proba) 지원 모델이 없습니다.")
            
    else:
        st.warning("4페이지에서 모형을 먼저 학습해주세요.")
