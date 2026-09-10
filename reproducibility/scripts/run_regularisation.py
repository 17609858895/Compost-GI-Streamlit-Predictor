"""Original 11-candidate experiment recovered from the 2026-05-31 execution record.

Estimator seeds vary with split seeds in this separate experiment.
Only the output path was made portable.
"""
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import generate_compost_figures as g
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

df=g.load_compost_data()
data=df[df[g.TARGET].notna()].copy().reset_index(drop=True)
y=data[g.TARGET].astype(float)
features=g.FEATURE_SETS['P1']
seeds=g.SEEDS
candidates={
 'XGB_current': xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.9, learning_rate=0.05360192649435464, max_depth=6, min_child_weight=3, n_estimators=150, subsample=0.7431844659980276, reg_lambda=3.0, random_state=1, n_jobs=-1, verbosity=0),
 'XGB_reg1': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=120, learning_rate=0.045, max_depth=4, min_child_weight=8, subsample=0.78, colsample_bytree=0.78, reg_lambda=8, reg_alpha=0.5, random_state=1, n_jobs=-1, verbosity=0),
 'XGB_reg2': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=90, learning_rate=0.05, max_depth=3, min_child_weight=10, subsample=0.75, colsample_bytree=0.75, reg_lambda=12, reg_alpha=1.0, random_state=1, n_jobs=-1, verbosity=0),
 'XGB_reg3': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=140, learning_rate=0.035, max_depth=3, min_child_weight=12, subsample=0.70, colsample_bytree=0.70, reg_lambda=15, reg_alpha=1.5, random_state=1, n_jobs=-1, verbosity=0),
 'HGB_reg1': HistGradientBoostingRegressor(max_iter=180, learning_rate=0.045, max_leaf_nodes=15, min_samples_leaf=55, l2_regularization=6.0, random_state=1),
 'HGB_reg2': HistGradientBoostingRegressor(max_iter=140, learning_rate=0.045, max_leaf_nodes=11, min_samples_leaf=75, l2_regularization=10.0, random_state=1),
 'LGBM_reg1': lgb.LGBMRegressor(n_estimators=180, learning_rate=0.035, num_leaves=11, max_depth=4, min_child_samples=60, subsample=0.75, colsample_bytree=0.75, reg_lambda=10, reg_alpha=1.0, random_state=1, n_jobs=-1, verbose=-1),
 'Cat_reg_current': cb.CatBoostRegressor(iterations=260, learning_rate=0.045, depth=5, l2_leaf_reg=8.0, loss_function='RMSE', random_seed=1, verbose=False, allow_writing_files=False),
 'Cat_reg2': cb.CatBoostRegressor(iterations=180, learning_rate=0.04, depth=4, l2_leaf_reg=14.0, random_strength=1.5, bagging_temperature=1.0, loss_function='RMSE', random_seed=1, verbose=False, allow_writing_files=False),
 'RF_reg': RandomForestRegressor(n_estimators=300, max_depth=9, min_samples_leaf=18, max_features=0.65, random_state=1, n_jobs=-1),
 'ET_reg': ExtraTreesRegressor(n_estimators=300, max_depth=9, min_samples_leaf=18, max_features=0.65, random_state=1, n_jobs=-1),
}
rows=[]
for name, est in candidates.items():
    for validation in ['group','random']:
        for seed in seeds:
            tr,te=g.split_indices(data,y,validation,seed,test_size=0.25)
            pipe=Pipeline([('pre', g.make_preprocessor(features, scale=False)), ('model', est.__class__(**est.get_params()))])
            # update random_state per seed if available
            params={}
            for p in ['random_state','random_seed']:
                if p in pipe.named_steps['model'].get_params(): params[p]=seed
            if params: pipe.named_steps['model'].set_params(**params)
            pipe.fit(data.iloc[tr][features], y.iloc[tr])
            ptr=pipe.predict(data.iloc[tr][features]); pte=pipe.predict(data.iloc[te][features])
            rows.append({'candidate':name,'validation':validation,'seed':seed,'train_r2':r2_score(y.iloc[tr],ptr),'test_r2':r2_score(y.iloc[te],pte),'gap':r2_score(y.iloc[tr],ptr)-r2_score(y.iloc[te],pte),'rmse':mean_squared_error(y.iloc[te],pte,squared=False),'mae':mean_absolute_error(y.iloc[te],pte)})
res=pd.DataFrame(rows)
sumry=res.groupby(['candidate','validation']).agg(train=('train_r2','mean'),test=('test_r2','mean'),gap=('gap','mean'),gap_max=('gap','max'),rmse=('rmse','mean'),test_sd=('test_r2','std')).round(3).reset_index()
print(sumry.sort_values(['validation','test'], ascending=[True,False]).to_string(index=False))
out=g.DATA_OUT / 'overfit_reduction_candidates.csv'
out.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(out,index=False,encoding='utf-8-sig')
print('saved',out)
