from pathlib import Path
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
ROOT=Path(__file__).resolve().parents[1]
def main():
 o=pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts']); c=pd.read_csv(ROOT/'data/clean/customers.csv')
 done=o[o.status.eq('completed')].copy(); snap=pd.Timestamp('2025-10-01')
 rfm=done.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
 rfm['recency']=(snap-rfm.last_order).dt.days; rfm=rfm.drop(columns='last_order')
 scaler=StandardScaler(); X=scaler.fit_transform(rfm[['recency','frequency','monetary']])
 sweep=[]
 for k in range(2,9):
  model=KMeans(n_clusters=k,n_init=10,random_state=42); labels=model.fit_predict(X)
  sweep.append({'k':k,'inertia':model.inertia_,'silhouette':silhouette_score(X,labels)})
 sweep_df=pd.DataFrame(sweep)
 print('\nPart B: K-means inertia and silhouette sweep')
 print(sweep_df.round({'inertia':1,'silhouette':3}).to_string(index=False))
 (ROOT/'outputs').mkdir(exist_ok=True)
 sweep_df.to_csv(ROOT/'outputs/segment_k_sweep.csv',index=False)
 model=KMeans(n_clusters=4,n_init=10,random_state=42); rfm['segment_id']=model.fit_predict(X)
 prof=rfm.groupby('segment_id').agg(customers=('customer_id','size'),recency=('recency','mean'),frequency=('frequency','mean'),monetary=('monetary','mean')).sort_values('monetary',ascending=False)
 print('\nSegment profile\n',prof.round(2).to_string())
 # Names are assigned by profile, not arbitrary cluster numeric ids.
 order=prof.index.tolist(); names={order[0]:'Premium Spenders',order[1]:'Frequent Shoppers',order[2]:'Occasional Shoppers',order[3]:'At-Risk Low Spend'}
 rfm['segment_name']=rfm.segment_id.map(names); rfm['is_premium']=rfm.segment_id.eq(order[0])
 pca=PCA(n_components=2,random_state=42).fit(X); print('PCA explained variance',pca.explained_variance_ratio_.round(3).tolist(),'total',round(pca.explained_variance_ratio_.sum(),3))
 out=rfm.merge(c[['customer_id']],on='customer_id',how='left'); out=out[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']]
 (ROOT/'outputs').mkdir(exist_ok=True); out.to_csv(ROOT/'outputs/customer_segments.csv',index=False)
 print('Saved outputs/customer_segments.csv rows=',len(out))
if __name__=='__main__':main()
