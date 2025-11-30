# Прогноз дохода клиента

Решение для хакатона Альфа-Банк Hack&Change 2025.

## Структура

```
alfabank-income-prediction/
├── data/
│   └── processed_data_here__google_drive_link_.txt
├── models/
│   └── trained_models_here__google_drive_link_.txt
├── demonstration_video_here__google_drive_link_.txt
├── 1_SHAP_Analysis.py
├── app.py
├── prepare_models.py
└── requirements.txt
```

`app.py` - главная страница  
`1_SHAP_Analysis.py` - SHAP анализ  
`prepare_models.py` - обучение модели  
`requirements.txt` - зависимости  

Данные: https://drive.google.com/drive/folders/1iq-BTe-AIxSe0J3pDYag63xbdZs1iZBU?usp=sharing  
Модели: https://drive.google.com/drive/folders/15CdR4BLFY0dYoO0fypC6RiqXDRoFOoWU?usp=sharing  
Видео: https://drive.google.com/drive/folders/1vh4_cslRhw_gGzjA4ecsnbpwYBzZfh4G?usp=sharing

## Требования

Python 3.10+  
4 GB RAM

## Установка

```bash
pip install -r requirements.txt
```

Зависимости: `streamlit==1.28.0`, `pandas==2.0.3`, `numpy==1.24.3`, `lightgbm==4.1.0`, `scikit-learn==1.3.0`, `joblib==1.3.2`, `shap==0.43.0`, `matplotlib==3.7.2`

## Запуск

Вариант 1 (с готовыми моделями):

```bash
# Скачать модели и данные по ссылкам выше
# Положить в папки models/ и data/
streamlit run app.py
```

Вариант 2 (обучение с нуля):

```bash
# Положить hackathon_income_train.csv и hackathon_income_test.csv в корень
python prepare_models.py  # 15-30 минут
streamlit run app.py
```

Адрес: `http://localhost:8501`

## Функционал

Главная страница: выбор клиента, прогноз дохода, сегментация, рекомендации продуктов, кредитный лимит  
SHAP анализ: объяснение предсказаний, waterfall график

## Модель

Алгоритм: LightGBM  
Архитектура: ансамбль 2 конфигурации × 5 folds = 10 моделей  
Признаков: 405 из 224 исходных  
Метрика: WMAE* = 37,218

Конфигурация 1: `num_leaves=63`, `lr=0.03`, `feature_fraction=0.7`, `lambda_l1=0.1`, `lambda_l2=0.1`  
Конфигурация 2: `num_leaves=31`, `lr=0.05`, `feature_fraction=0.8`

## Сегментация

```
Базовый:     < 30k    руб/мес  →  лимит доход×3  (до 50k)   →  дебетовая карта, микрокредит
Стандарт:    30-80k   руб/мес  →  лимит доход×6  (до 300k)  →  кредитная карта, потреб. кредит
Премиум:     80-150k  руб/мес  →  лимит доход×8  (до 700k)  →  премиальная карта, инвестиции
Привилегия:  > 150k   руб/мес  →  лимит доход×12 (до 2M)    →  private banking, ипотека
```

## Команда

Bugging Bad

Мазилкин Никита (лидер, Data Scientist)  
Клементьева Виктория (Data Analyst)  
Климентенко Сергей (Data Scientist)  
Павленко Ульяна (Backend)  
Сафин Роман (Data Analyst)

Альфа-Банк Hack&Change 2025
