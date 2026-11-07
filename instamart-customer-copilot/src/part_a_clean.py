"""Part A: audit, clean, persist and validate the synthetic Instamart data."""
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW, CLEAN = ROOT / "data/raw", ROOT / "data/clean"
CLEAN.mkdir(parents=True, exist_ok=True)

def audit(df, name):
    print(f"\nAUDIT: {name} shape={df.shape}")
    print("dtypes:\n", df.dtypes.to_string())
    missing = pd.DataFrame({"missing_count": df.isna().sum(), "missing_pct": (df.isna().mean()*100).round(2)})
    print("missing by column:\n", missing.to_string())

def main():
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv")
    for df, name in [(customers,"customers"),(products,"products"),(orders,"orders")]: audit(df,name)
    print("\nRAW CHECKS", {
        "orders_rows":len(orders), "exact_duplicates":int(orders.duplicated().sum()),
        "pending":int(orders.status.eq('pending').sum()), "distinct_order_id":int(orders.order_id.nunique()),
        "non_numeric_amount":int(pd.to_numeric(orders.order_amount,errors='coerce').isna().sum()),
        "negative_delivery":int(pd.to_numeric(orders.delivery_minutes,errors='coerce').lt(0).sum()),
        "blank_delivery":int((orders.delivery_minutes.isna() | orders.delivery_minutes.astype(str).str.strip().eq('')).sum()),
        "blank_age":int((customers.age.isna() | customers.age.astype(str).str.strip().eq('')).sum()),
        "age_outside_10_100":int(pd.to_numeric(customers.age,errors='coerce').lt(10).fillna(False).sum()+pd.to_numeric(customers.age,errors='coerce').gt(100).fillna(False).sum()),
        "raw_city_spellings":int(customers.city.nunique()),
        "raw_gender_spellings_nonblank":int(customers.loc[customers.gender.astype(str).str.strip().ne(''),'gender'].nunique()),
        "blank_gender":int((customers.gender.isna() | customers.gender.astype(str).str.strip().eq('')).sum())})
    # Customers
    customers['city'] = customers.city.astype(str).str.strip().str.title()
    customers['gender'] = customers.gender.astype(str).str.strip().str.lower().map({'m':'male','male':'male','f':'female','female':'female'}).fillna('unknown')
    customers['age'] = pd.to_numeric(customers.age, errors='coerce')
    customers.loc[~customers.age.between(10,100), 'age'] = np.nan
    customers['age'] = customers['age'].fillna(customers.groupby('city')['age'].transform('median'))
    customers['age'] = customers['age'].fillna(customers.age.median()).round().astype(int)
    customers['signup_date'] = pd.to_datetime(customers.signup_date, errors='coerce')
    # Products
    products['price_inr'] = pd.to_numeric(products.price_inr, errors='raise')
    # Orders: order of operations is intentional.
    orders['order_ts'] = pd.to_datetime(orders.order_ts, errors='coerce')
    orders = orders.drop_duplicates()
    orders = orders.sort_values(['order_id','order_ts']).drop_duplicates('order_id', keep='last').copy()
    orders['quantity'] = pd.to_numeric(orders.quantity, errors='coerce')
    orders['discount_pct'] = pd.to_numeric(orders.discount_pct, errors='coerce').fillna(0)
    orders['order_amount'] = pd.to_numeric(orders.order_amount, errors='coerce')
    price = products.set_index('product_id').price_inr
    fallback = orders.product_id.map(price) * orders.quantity * (1 - orders.discount_pct/100)
    orders['order_amount'] = orders.order_amount.fillna(fallback).round(2)
    orders['delivery_minutes'] = pd.to_numeric(orders.delivery_minutes, errors='coerce')
    orders.loc[orders.delivery_minutes < 0, 'delivery_minutes'] = np.nan
    category_map = products.set_index('product_id').category
    orders['category'] = orders.product_id.map(category_map)
    medians = orders.loc[orders.status.eq('completed')].groupby('category').delivery_minutes.median()
    mask = orders.status.eq('completed') & orders.delivery_minutes.isna()
    orders.loc[mask,'delivery_minutes'] = orders.loc[mask,'category'].map(medians)
    orders = orders.drop(columns=['category'])
    # Keep SQLite and CSV date formats reproducible.
    customers.to_csv(CLEAN/'customers.csv', index=False, date_format='%Y-%m-%d')
    products.to_csv(CLEAN/'products.csv', index=False)
    orders.to_csv(CLEAN/'orders.csv', index=False, date_format='%Y-%m-%d %H:%M:%S')
    db = ROOT/'data/instamart.db'; db.parent.mkdir(exist_ok=True)
    with sqlite3.connect(db) as conn:
        customers_db=customers.copy(); customers_db['signup_date']=customers_db.signup_date.dt.strftime('%Y-%m-%d')
        orders_db=orders.copy(); orders_db['order_ts']=orders_db.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')
        customers_db.to_sql('customers',conn,if_exists='replace',index=False)
        products.to_sql('products',conn,if_exists='replace',index=False)
        orders_db.to_sql('orders',conn,if_exists='replace',index=False)
    print('\nCLEAN CHECKS')
    print('customers cities:',customers.city.value_counts().to_dict())
    print('customers gender:',customers.gender.value_counts().to_dict())
    print('orders status:',orders.status.value_counts().to_dict())
    print('completed revenue INR:',round(orders.loc[orders.status.eq('completed'),'order_amount'].sum(),2))
    print('orders rows:',len(orders),'missing completed delivery:',int(orders.loc[orders.status.eq('completed'),'delivery_minutes'].isna().sum()),'cancelled missing delivery:',int(orders.loc[orders.status.eq('cancelled'),'delivery_minutes'].isna().sum()))
if __name__ == '__main__': main()
