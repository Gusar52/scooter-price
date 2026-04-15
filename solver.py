import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

import lightgbm as lgb
from category_encoders import TargetEncoder

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


def create_features(df):
    df = df.copy()

    df['speed'] = df['distance_km'] / (df['trip_duration_min'] / 60)
    df['distance_per_min'] = df['distance_km'] / df['trip_duration_min']

    df['battery_per_km'] = df['battery_level_start'] / (df['distance_km'] + 1e-5)

    df['distance_diff'] = abs(df['distance_km'] - df['distance_km_noisy'])

    df['temp_x_demand'] = df['temperature_c'] * df['demand_index']

    df['complexity_x_distance'] = df['route_complexity'] * df['distance_km']

    df['price_x_demand'] = df['avg_price_last_week'] * df['demand_index']
    df['price_x_distance'] = df['avg_price_last_week'] * df['distance_km']
    df['demand_x_distance'] = df['demand_index'] * df['distance_km']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


train = create_features(train)
test = create_features(test)

y = train['rental_price']

X = train.drop(['rental_price', 'id'], axis=1)
X_test = test.drop(['id'], axis=1)

categorical_cols = ['city_zone', 'scooter_model']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n🔥 Fold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    encoder = TargetEncoder(cols=categorical_cols, smoothing=10)
    X_train = encoder.fit_transform(X_train, y_train)
    X_val = encoder.transform(X_val)
    X_test_enc = encoder.transform(X_test)


    X_train = X_train.fillna(X_train.median())
    X_val = X_val.fillna(X_train.median())
    X_test_enc = X_test_enc.fillna(X_train.median())
    model_lgb = lgb.LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.015,
        num_leaves=50,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.2,
        min_child_samples=20,
        random_state=42
    )

    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    val_pred_lgb = model_lgb.predict(X_val)
    test_pred_lgb = model_lgb.predict(X_test_enc)

    model_lin = Ridge(alpha=1.0)
    model_lin.fit(X_train, y_train)

    val_pred_lin = model_lin.predict(X_val)
    test_pred_lin = model_lin.predict(X_test_enc)

    val_pred = 0.8 * val_pred_lgb + 0.2 * val_pred_lin
    test_pred = 0.8 * test_pred_lgb + 0.2 * test_pred_lin

    oof_preds[val_idx] = val_pred
    test_preds += test_pred / kf.n_splits

score = r2_score(y, oof_preds)
print(f"R2: {score:.4f}")

test_preds = np.clip(test_preds, 0, None)

submission = pd.DataFrame({
    'id': test['id'],
    'rental_price': test_preds
})

submission.to_csv('submission.csv', index=False)