import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Прогноз дохода клиента",
    layout="wide"
)

@st.cache_data
def load_data():
    try:
        test_df = pd.read_csv('data/test_processed.csv')
        return test_df
    except FileNotFoundError:
        st.error("Файл data/test_processed.csv не найден")
        return None

@st.cache_resource
def load_model():
    try:
        models = joblib.load('models/ensemble_models.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        return models, feature_cols
    except FileNotFoundError:
        st.error("Модели не найдены")
        return None, None

def generate_recommendations(predicted_income):
    if predicted_income < 30000:
        segment = "Базовый"
        products = ["Дебетовая карта", "Микрокредит"]
        credit_limit = min(predicted_income * 3, 50000)
    elif predicted_income < 80000:
        segment = "Стандарт"
        products = ["Кредитная карта", "Потребительский кредит"]
        credit_limit = min(predicted_income * 6, 300000)
    elif predicted_income < 150000:
        segment = "Премиум"
        products = ["Премиальная карта", "Инвестиционные продукты"]
        credit_limit = min(predicted_income * 8, 700000)
    else:
        segment = "Привилегия"
        products = ["Private Banking", "Ипотека"]
        credit_limit = min(predicted_income * 12, 2000000)
    
    return {
        "segment": segment,
        "products": products,
        "credit_limit": credit_limit
    }

def main():
    st.title("Прогноз дохода клиента")
    st.markdown("---")
    
    test_df = load_data()
    models, feature_cols = load_model()
    
    if test_df is None or models is None:
        st.code("python prepare_models.py")
        return
    
    st.sidebar.header("Параметры")
    
    client_id = st.sidebar.selectbox(
        "ID клиента",
        options=test_df['id'].head(100).tolist()
    )
    
    if st.sidebar.button("Прогноз"):
        
        client_data = test_df[test_df['id'] == client_id].iloc[0]
        X_client = client_data[feature_cols].values.reshape(1, -1)
        
        predictions = []
        for model_set in models:
            pred = np.mean([m.predict(X_client)[0] for m in model_set['models']])
            predictions.append(pred)
        
        final_prediction = np.mean(predictions)
        recommendations = generate_recommendations(final_prediction)
        
        st.subheader(f"Прогноз: {final_prediction:,.0f} руб/мес")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Сегмент", recommendations['segment'])
            st.metric("Кредитный лимит", f"{recommendations['credit_limit']:,.0f} руб")
        
        with col2:
            st.write("**Продукты:**")
            for product in recommendations['products']:
                st.write(f"- {product}")

if __name__ == "__main__":
    main()
