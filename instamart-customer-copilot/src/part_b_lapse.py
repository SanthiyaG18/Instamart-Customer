from pathlib import Path
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,f1_score,precision_score,recall_score
ROOT=Path(__file__).resolve().parents[1]
def main():
 o=pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts']); c=pd.read_csv(ROOT/'data/clean/customers.csv',parse_dates=['signup_date']); p=pd.read_csv(ROOT/'data/clean/products.csv')
 cutoff=pd.Timestamp('2025-09-01'); pre=o[(o.order_ts<cutoff)].copy(); completed=pre[pre.status.eq('completed')]
 last=completed.groupby('customer_id').order_ts.max(); cohort=sorted(last.index)
 post=o[(o.order_ts>=cutoff)&(o.order_ts<pd.Timestamp('2025-10-01'))&o.status.eq('completed')].groupby('customer_id').size()
 y=pd.Series([int(cid not in post.index) for cid in cohort],index=cohort,name='lapsed')
 f=completed.groupby('customer_id').agg(frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean'))
 rec=(cutoff-last).dt.days.rename('recency'); canc=pre[pre.status.eq('cancelled')].groupby('customer_id').size(); total=pre.groupby('customer_id').size(); cancel_rate=(canc.reindex(cohort,fill_value=0)/total.reindex(cohort)).rename('cancellation_rate')
 category_counts=(completed.merge(p[['product_id','category']],on='product_id')
                  .groupby(['customer_id','category']).size().rename('count').reset_index())
 fav=(category_counts.sort_values(['customer_id','count','category'],ascending=[True,False,True])
      .drop_duplicates('customer_id').set_index('customer_id')['category'].rename('favourite_category'))
 features=pd.concat([f,rec,cancel_rate,fav],axis=1).reindex(cohort).reset_index().rename(columns={'index':'customer_id'})
 features=features.merge(c[['customer_id','city','age','gender','signup_date']],on='customer_id',how='left'); features['tenure_days']=(cutoff-features.signup_date).dt.days; features=features.drop(columns='signup_date')
 X=features.drop(columns='customer_id'); y=y.reindex(features.customer_id).astype(int)
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
 cat=X.select_dtypes(include='object').columns.tolist(); num=[x for x in X.columns if x not in cat]
 prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),cat)])
 log=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000))]); gs=GridSearchCV(log,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1); gs.fit(Xtr,ytr)
 rf=Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=100,random_state=42))]); gr=GridSearchCV(rf,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1); gr.fit(Xtr,ytr)
 for name,m in [('Logistic',gs.best_estimator_),('RandomForest',gr.best_estimator_)]:
  pred=m.predict(Xte); print('\n',name,'params',gs.best_params_ if name=='Logistic' else gr.best_params_); print(classification_report(yte,pred,zero_division=0)); print('confusion',confusion_matrix(yte,pred).tolist())
 best=gr.best_estimator_ if f1_score(yte,gr.best_estimator_.predict(Xte))>=f1_score(yte,gs.best_estimator_.predict(Xte)) else gs.best_estimator_
 joblib.dump(best,ROOT/'outputs/lapse_model.joblib'); assert np.array_equal(best.predict(Xte),joblib.load(ROOT/'outputs/lapse_model.joblib').predict(Xte))
 # Oversample training rows only, then refit the selected estimator configuration.
 train=pd.concat([Xtr.assign(_y=ytr.values),Xtr.assign(_y=ytr.values).loc[ytr.eq(1).values].sample(n=max(0,int((ytr==0).sum()-(ytr==1).sum())),replace=True,random_state=42)]) if (ytr==1).sum()<(ytr==0).sum() else Xtr.assign(_y=ytr.values)
 overs=best.__class__(best.steps) if False else best
 # Refit a cloned pipeline on oversampled data to avoid changing the saved best model.
 from sklearn.base import clone
 overs=clone(best); overs.fit(train.drop(columns='_y'),train['_y'])
 pred=overs.predict(Xte); print('Oversampled test precision/recall/F1:',round(precision_score(yte,pred,zero_division=0),3),round(recall_score(yte,pred,zero_division=0),3),round(f1_score(yte,pred,zero_division=0),3))
 # Feature importance for forest; logistic coefficients are also printed when available.
 rfbest=gr.best_estimator_; model=rfbest.named_steps['model']; names=rfbest.named_steps['prep'].get_feature_names_out(); imp=pd.Series(model.feature_importances_,index=names).sort_values(ascending=False).head(5); print('Top RF importances\n',imp.to_string())
 log_model=gs.best_estimator_
 coefs=pd.Series(log_model.named_steps['model'].coef_[0],index=log_model.named_steps['prep'].get_feature_names_out()).sort_values(); print('Logistic coefficient signs (lowest/highest)\n',pd.concat([coefs.head(3),coefs.tail(3)]).to_string())
 # Retention list from customer segment Premium; this is a prioritization list, not a test metric.
 seg=pd.read_csv(ROOT/'outputs/customer_segments.csv'); premium=seg[seg.is_premium.astype(str).str.lower().isin(['true','1'])][['customer_id']]
 probs=best.predict_proba(X)[:,list(best.named_steps['model'].classes_).index(1)]
 scores=pd.DataFrame({'customer_id':features.customer_id,'lapse_probability':probs,'actually_lapsed':y.values}).merge(premium,on='customer_id',how='inner').sort_values('lapse_probability',ascending=False).head(10)
 scores.to_csv(ROOT/'outputs/retention_list.csv',index=False); print('Cohort',len(y),'lapsed',int(y.sum()),'rate',round(y.mean(),4),'saved retention list',len(scores))
if __name__=='__main__':main()
