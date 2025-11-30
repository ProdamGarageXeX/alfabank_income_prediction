import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="SHAP анализ",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        models = joblib.load('models/ensemble_models.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        return models, feature_cols
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_data():
    try:
        test_df = pd.read_csv('data/test_processed.csv')
        return test_df
    except FileNotFoundError:
        return None

def compute_shap_values(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values

def main():
    st.title("SHAP анализ")
    st.markdown("---")
    
    models, feature_cols = load_model()
    test_df = load_data()
    
    if models is None or test_df is None:
        st.error("Модели не найдены")
        return
    
    st.sidebar.header("Параметры")
    
    client_id = st.sidebar.selectbox(
        "ID клиента",
        options=test_df['id'].head(50).tolist()
    )
    
    max_display = st.sidebar.slider(
        "Количество признаков",
        min_value=5,
        max_value=30,
        value=15
    )
    
    if st.sidebar.button("Вычислить"):
        
        with st.spinner("Вычисление..."):
            
            client_data = test_df[test_df['id'] == client_id]
            X_client = client_data[feature_cols].values
            
            model = models[0]['models'][0]
            explainer, shap_values = compute_shap_values(model, X_client)
            prediction = model.predict(X_client)[0]
            
            st.success("Готово")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Прогноз", f"{prediction:,.0f} руб")
            
            with col2:
                st.metric("Base value", f"{explainer.expected_value:,.0f} руб")
            
            st.markdown("---")
            
            st.subheader("Waterfall Plot")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=X_client[0],
                    feature_names=feature_cols
                ),
                max_display=max_display,
                show=False
            )
            
            st.pyplot(fig)
            plt.close()

if __name__ == "__main__":
    main()
