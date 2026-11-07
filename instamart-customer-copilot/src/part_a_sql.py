from pathlib import Path
import sqlite3, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 db=ROOT/'data/instamart.db'; sql=(ROOT/'sql/queries.sql').read_text(); statements=[x.strip() for x in sql.split(';') if x.strip()]
 with sqlite3.connect(db) as con:
  for i,q in enumerate(statements,1):
   df=pd.read_sql_query(q,con); print(f'\nQ{i}: rows={len(df)}'); print(df.head(8).to_string(index=False))
  o=pd.read_sql_query("SELECT o.*,p.category,c.city FROM orders o JOIN products p USING(product_id) JOIN customers c USING(customer_id) WHERE o.status='completed'",con)
  c=pd.read_sql_query('SELECT * FROM customers',con)
 pivot=pd.pivot_table(o,index='city',columns='category',values='order_amount',aggfunc='sum',fill_value=0,margins=True)
 print('Pandas pivot Q2 grand total INR:',round(float(o['order_amount'].sum()),2))
 # Required city statistics from joined table.
 with sqlite3.connect(db) as con:
  stats=pd.read_sql_query("SELECT c.city,ROUND(SUM(o.order_amount),2) AS sum_inr,ROUND(AVG(o.order_amount),2) AS mean_inr,COUNT(*) AS order_count FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.city ORDER BY c.city",con)
 print('\nCITY STATS\n',stats.to_string(index=False))
 x=o.order_amount; q1,q3=x.quantile(.25),x.quantile(.75); iqr=q3-q1; lo,hi=q1-1.5*iqr,q3+1.5*iqr
 print('\nAMOUNT STATS',{'mean':round(x.mean(),2),'median':round(x.median(),2),'skew':round(x.skew(),2),'q1':round(q1,2),'q3':round(q3,2),'lower_fence':round(lo,2),'upper_fence':round(hi,2),'below':int((x<lo).sum()),'above':int((x>hi).sum())})
 print('\nCORRELATION\n',o[['order_amount','discount_pct','quantity','delivery_minutes']].corr().round(3).to_string())
if __name__=='__main__':main()
