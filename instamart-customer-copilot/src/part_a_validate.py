from pathlib import Path
import pandas as pd, sqlite3
ROOT=Path(__file__).resolve().parents[1]
def main():
 c=pd.read_csv(ROOT/'data/clean/customers.csv'); p=pd.read_csv(ROOT/'data/clean/products.csv'); o=pd.read_csv(ROOT/'data/clean/orders.csv')
 checks=[len(c)==1200,len(p)==40,len(o)==12309,o.order_id.nunique()==12309,not o.status.eq('pending').any(),o.status.value_counts().to_dict()=={'completed':11110,'cancelled':1199},int(c.age.isna().sum())==0,set(c.city)=={'Pune','Delhi','Bengaluru','Chennai','Hyderabad','Mumbai'},set(c.gender)=={'female','male','unknown'},round(o.loc[o.status.eq('completed'),'order_amount'].sum(),2)==2998668.80,int(o.loc[o.status.eq('completed'),'delivery_minutes'].isna().sum())==0,int(o.loc[o.status.eq('cancelled'),'delivery_minutes'].isna().sum())==1199]
 with sqlite3.connect(ROOT/'data/instamart.db') as con:
  tables={r[0] for r in con.execute("select name from sqlite_master where type='table'")}
  checks.append({'customers','products','orders'}.issubset(tables))
 if not all(checks):
  for i,v in enumerate(checks,1): print(f'CHECK {i}: {v}')
  raise AssertionError('One or more Part A checkpoints failed')
 print('ALL CHECKPOINTS PASSED')
if __name__=='__main__':main()
