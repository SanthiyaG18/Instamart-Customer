"""Privacy-first ticket classification with retry, validation, budget and routing."""
from pathlib import Path
import sys, re, json, random, time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import mock_llm
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
ALLOWED={'billing','delivery','technical','fraud'}
PATTERNS=[('email',re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')),
          ('phone',re.compile(r'\b[6-9]\d{9}\b')),
          ('order_id',re.compile(r'ORD-\d{6}'))]
def redact(text):
 counts={}
 for name,pat in PATTERNS:
  text,n=pat.subn({'email':'[EMAIL]','phone':'[PHONE]','order_id':'[ORDER_ID]'}[name],text); counts[name]=n
 return text,counts
def call_with_retry(messages,max_retries=3):
 for attempt in range(max_retries+1):
  try: return mock_llm.chat(messages),attempt
  except mock_llm.TransientLLMError:
   if attempt>=max_retries: raise
   delay=.05*(2**attempt)*random.uniform(.9,1.15); print('retry',attempt+1,'wait',round(delay,3)); time.sleep(delay)
def classify(text,system,budget_left):
 messages=[{'role':'system','content':system},{'role':'user','content':text}]
 estimated=mock_llm.est_tokens(' '.join(m['content'] for m in messages))
 if estimated>budget_left: return None,0,'budget_exhausted'
 response,retries=call_with_retry(messages)
 msg=response.choices[0].message; raw=msg.content or ''; used=response.usage.total_tokens
 try:
  parsed=json.loads(raw); cat=parsed.get('category')
  if cat in ALLOWED: return cat,used,'classified_json'
  return None,used,'flagged_invalid_output'
 except (json.JSONDecodeError,TypeError):
  words=re.findall(r'[a-z]+',raw.lower()); matches=[w for w in words if w in ALLOWED]
  if len(set(matches))==1: return matches[0],used,'classified_keyword'
  return None,used,'flagged_invalid_output'
def route(status,category,segment):
 if status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}: return 'human_review','NORMAL'
 if category=='fraud': return 'security_team','HIGH'
 if segment=='Premium Spenders': return 'premium_desk','HIGH'
 return 'support_queue','NORMAL'
def run_batch(budget, few_shot, filename):
 tickets=pd.read_csv(ROOT/'data/raw/tickets.csv',keep_default_na=False)
 seg=pd.read_csv(OUT/'customer_segments.csv')
 segmap=seg.set_index('customer_id').segment_name.to_dict()
 system='CLASSIFY. Classify into exactly one category: billing, delivery, technical, fraud. Return JSON only with keys category and reason.'
 if few_shot: system+='\nEXAMPLES:\nUnauthorized card transaction -> fraud\nThe app crashes at login -> technical\nMy delivery is late -> delivery'
 mock_llm.reset(); random.seed(42); used_total=0; rows=[]
 for _,t in tickets.iterrows():
  tid,cid,txt=t.ticket_id,t.customer_id,t.text; segment=segmap.get(cid,'')
  if not str(txt).strip(): status,cat,used='skipped_empty',None,0
  else:
   safe,counts=redact(str(txt))
   assert not any(p.search(safe) for _,p in PATTERNS),f'PII remains in {tid}'
   if mock_llm.est_tokens(safe)>400: status,cat,used='flagged_long',None,0
   elif used_total>=budget: status,cat,used='budget_exhausted',None,0
   else:
    cat,used,status=classify(safe,system,budget-used_total)
    used_total+=used
  route_to,priority=route(status,cat,segment)
  rows.append({'ticket_id':tid,'customer_id':cid,'status':status,'category':cat or '', 'segment_name':segment,'priority':priority,'route':route_to,'tokens_used':used})
 df=pd.DataFrame(rows); df.to_csv(OUT/filename,index=False)
 eligible=df[df.status.isin(['classified_json','classified_keyword'])]
 truth=tickets.set_index('ticket_id').true_category
 correct=(eligible.category==eligible.ticket_id.map(truth)).sum()
 print(filename,'rows',len(df),'status counts',df.status.value_counts().to_dict(),'tokens',used_total,'classified accuracy',round(correct/max(1,len(eligible)),4))
 if not few_shot:
  from sklearn.metrics import confusion_matrix
  print('zero-shot confusion matrix categories',sorted(ALLOWED)); print(confusion_matrix(eligible.ticket_id.map(truth),eligible.category,labels=sorted(ALLOWED)))
def main():
 run_batch(5000,False,'triage_budget_demo.csv')
 # Full zero-shot evaluation uses a generous budget so all eligible tickets are measured.
 run_batch(100000,False,'triage_zero_shot_eval.csv')
 run_batch(100000,True,'triage_results.csv')
if __name__=='__main__':main()
