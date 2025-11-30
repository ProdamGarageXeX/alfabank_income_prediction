import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Директории
Path('models').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)

def wmae_score(y_true, y_pred, weights):
    # Метрика WMAE
    return np.mean(weights * np.abs(y_true - y_pred))

def advanced_feature_engineering(df, is_train=True):
    df = df.copy()
    
    # 1. Конвертация object -> numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 2. Логарифмирование денежных признаков
    money_keywords = ['sum', 'amt', 'turn', 'limit', 'salary', 'income', 'avg', 'max', 'payout']
    money_cols = [c for c in df.columns if any(kw in c.lower() for kw in money_keywords)]
    for col in money_cols:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0).fillna(0))
    
    # 3. Отношения между признаками
    ratio_pairs = [
        ('turn_cur_cr_sum_v2', 'turn_cur_db_sum_v2'),
        ('turn_cur_cr_avg_v2', 'turn_cur_db_avg_v2'),
        ('turn_cur_cr_max_v2', 'turn_cur_db_max_v2'),
        ('avg_cur_cr_turn', 'avg_cur_db_turn'),
        ('turn_cur_cr_avg_v2', 'turn_cur_cr_max_v2'),
    ]
    for num_col, den_col in ratio_pairs:
        if num_col in df.columns and den_col in df.columns:
            df[f'ratio_{num_col[:15]}_{den_col[:15]}'] = df[num_col] / (df[den_col] + 1e-8)
    
    # 4. Разности
    if 'turn_cur_cr_sum_v2' in df.columns and 'turn_cur_db_sum_v2' in df.columns:
        df['diff_cr_db_sum'] = df['turn_cur_cr_sum_v2'] - df['turn_cur_db_sum_v2']
        df['diff_cr_db_abs'] = np.abs(df['diff_cr_db_sum'])
    
    # 5. Агрегации по категориям
    category_groups = {
        'grocery': ['supermarkety', 'gipermarkety', 'produkty'],
        'transport': ['transport', 'taksi', 'avto', 'toplivo'],
        'entertainment': ['restorany', 'kino', 'razvlechenija'],
        'finance': ['bankomat', 'perevod', 'platezh', 'vydacha'],
        'health': ['apteki', 'medicina'],
        'shopping': ['odezhda', 'obuv'],
    }
    for group_name, keywords in category_groups.items():
        group_cols = [c for c in df.columns if any(kw in c.lower() for kw in keywords)]
        if group_cols:
            df[f'total_{group_name}'] = df[group_cols].fillna(0).sum(axis=1)
            df[f'mean_{group_name}'] = df[group_cols].fillna(0).mean(axis=1)
            df[f'std_{group_name}'] = df[group_cols].fillna(0).std(axis=1)
    
    # 6. БКИ агрегации
    bki_cols = [c for c in df.columns if 'bki' in c.lower()]
    if bki_cols:
        df['bki_mean'] = df[bki_cols].mean(axis=1)
        df['bki_std'] = df[bki_cols].std(axis=1)
        df['bki_max'] = df[bki_cols].max(axis=1)
        df['bki_sum'] = df[bki_cols].sum(axis=1)
    
    # 7. Цифровой профиль
    dp_cols = [c for c in df.columns if c.startswith('dp_')]
    if dp_cols:
        df['dp_mean'] = df[dp_cols].mean(axis=1)
        df['dp_std'] = df[dp_cols].std(axis=1)
        df['dp_max'] = df[dp_cols].max(axis=1)
    
    # 8. Индикаторы пропусков
    important_cols = ['salary_6to12m_avg', 'first_salary_income', 'incomeValue']
    for col in important_cols:
        if col in df.columns:
            df[f'{col}_isna'] = df[col].isna().astype(int)
    
    # 9. Категориальные признаки
    cat_cols = ['gender', 'adminarea', 'city_smart_name']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    return df

def train_models():
    # Загрузка данных
    print("\n[1/5] Загрузка данных")
    try:
        train_df = pd.read_csv('hackathon_income_train.csv', 
                              decimal=',', sep=';', low_memory=False)
        test_df = pd.read_csv('hackathon_income_test.csv',
                             decimal=',', sep=';', low_memory=False)
        print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    except FileNotFoundError:
        print("Файлы не найдены")
        return
    
    # Feature Engineering
    print("\n[2/5] Feature Engineering")
    train_fe = advanced_feature_engineering(train_df, is_train=True)
    test_fe = advanced_feature_engineering(test_df, is_train=False)
    
    # Выбор признаков
    exclude_cols = ['id', 'dt', 'target', 'w', 'incomeValue', 'incomeValueCategory']
    feature_cols = [c for c in train_fe.columns 
                   if c not in exclude_cols 
                   and train_fe[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    # Выравнивание колонок
    for col in feature_cols:
        if col not in test_fe.columns:
            test_fe[col] = 0
    
    print(f"Признаков: {len(feature_cols)}")
    
    # Подготовка данных
    X = train_fe[feature_cols].astype(np.float32).fillna(-999)
    y = train_fe['target'].values
    w = train_fe['w'].values
    X_test = test_fe[feature_cols].astype(np.float32).fillna(-999)
    
    # Сохранение обработанного теста для Streamlit
    test_processed = test_fe[['id'] + feature_cols].copy()
    test_processed.to_csv('data/test_processed.csv', index=False)
    print("Сохранен data/test_processed.csv")
    
    # Обучение моделей
    print("\n[3/5]")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Модель 1
    print("\nМодель 1 LightGBM оптимизированные параметры")
    params_lgb1 = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 30,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1
    }
    
    models_lgb1 = []
    oof_lgb1 = np.zeros(len(X))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold + 1}/5", end=" ")
        
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        w_tr, w_val = w[tr_idx], w[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data)
        
        model = lgb.train(
            params_lgb1,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=3000,
            callbacks=[
                lgb.early_stopping(150),
                lgb.log_evaluation(0)
            ]
        )
        
        pred = model.predict(X_val)
        oof_lgb1[val_idx] = pred
        models_lgb1.append(model)
        
        score = wmae_score(y_val, pred, w_val)
        print(f"WMAE*: {score:.0f}")
    
    oof_score1 = wmae_score(y, oof_lgb1, w)
    print(f"OOF WMAE*: {oof_score1:.2f}")
    
    # Модель 2
    print("\nМодель 2 LightGBM с альт параметрами)")
    params_lgb2 = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.05,
        'lambda_l2': 0.05,
        'verbose': -1,
        'seed': 43,
        'n_jobs': -1
    }
    
    models_lgb2 = []
    oof_lgb2 = np.zeros(len(X))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold + 1}/5", end=" ")
        
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        w_tr, w_val = w[tr_idx], w[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data)
        
        model = lgb.train(
            params_lgb2,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=3000,
            callbacks=[
                lgb.early_stopping(150),
                lgb.log_evaluation(0)
            ]
        )
        
        pred = model.predict(X_val)
        oof_lgb2[val_idx] = pred
        models_lgb2.append(model)
        
        score = wmae_score(y_val, pred, w_val)
        print(f"WMAE*: {score:.0f}")
    
    oof_score2 = wmae_score(y, oof_lgb2, w)
    print(f"OOF WMAE*: {oof_score2:.2f}")
    
    # Сохранение моделей
    print("\n[4/5]")
    
    ensemble_models = [
        {'name': 'lgb1', 'models': models_lgb1, 'weight': 0.5},
        {'name': 'lgb2', 'models': models_lgb2, 'weight': 0.5}
    ]
    
    joblib.dump(ensemble_models, 'models/ensemble_models.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    
    print("Сохранены:")
    print("  - models/ensemble_models.pkl")
    print("  - models/feature_columns.pkl")
    
    print(f"\nОбщая OOF WMAE*: {min(oof_score1, oof_score2):.2f}")

if __name__ == "__main__":
    train_models()
