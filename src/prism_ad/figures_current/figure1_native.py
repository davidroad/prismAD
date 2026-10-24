#!/usr/bin/env python3
"""Reorder the existing PRISM-AD workflow and show frozen clinical outcome branches."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.text import Text
from PIL import Image
import os
HERE=Path(__file__).resolve().parent
DATA=Path(os.environ['PRISM_AD_SOURCE_DATA'])
SOURCE=HERE/'architecture_template.py'
AGE=DATA/'plotting/age_summary.csv'
SUPPORT=DATA/'tables/first_visit_support.csv'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=SOURCE.read_text()
markers=['# a. Clinical targets','# b. Names and data types','# c. History also supplies','# d. Fitted preprocessing','# e. Term inspection','# Geometric verification']
for marker in markers:assert source.count(marker)==1
head=source.split(markers[0])[0]
env={'__file__':str(SOURCE),'__name__':'frozen_integrated_no_export'}
exec(compile(head,str(SOURCE),'exec'),env)
fig=env['fig'];ax=env['ax'];TEXT=env['TEXT'];BOXES=env['BOXES']
panel_transforms={}
# Retain every existing architecture/history/fitting/analysis artist. Only
# vertical placement, panel letters and frame allocation change.
for index,letter,origin,target,scale in [(1,'a',27,5,.9),(2,'b',150,117,27/31),(3,'c',185,149,36/41),(4,'e',230,236,13/17)]:
 block=markers[index]+source.split(markers[index])[1].split(markers[index+1])[0]
 previous=set(ax.get_children());old_box_n=len(BOXES)
 exec(compile(block,str(SOURCE),'exec'),env)
 trans=Affine2D().scale(1,scale).translate(0,target-scale*origin)+ax.transData
 for p in ax.get_children():
  if p not in previous:p.set_transform(trans)
 for t in TEXT:
  if t.get_position()==(5,origin) and t.get_fontweight()=='bold':t.set_text(letter)
 panel_transforms[letter]={'origin':origin,'target':target,'scale':scale,'module_indices':list(range(old_box_n,len(BOXES)))}
# Read only the registered aggregate files. No conversions are inferred.
age=pd.read_csv(AGE);support=pd.read_csv(SUPPORT)
counts={};source_rows=[];support_rows=[]
for task in ['CN_MCI','MCI_AD']:
 counts[task]={}
 for h in [3,5]:
  q=age[(age.task==task)&age.horizon.eq(h)]
  c={g:int(q.loc[q.group.eq(g),'n'].iloc[0]) if q.group.eq(g).any() else 0 for g in ['all','target','event_free','competing_ad','unknown']}
  assert c['all']==c['target']+c['event_free']+c['competing_ad']+c['unknown']
  s=support[(support.task==task)&support.horizon.eq(h)]
  assert len(s)==1
  r=s.iloc[0]
  assert int(r.n_risk)==int(r.n_subjects)==c['all']
  assert int(r.n_target)==c['target']
  assert int(r.n_known)==c['all']-c['unknown']
  counts[task][str(h)]=c;source_rows.extend(q.to_dict('records'));support_rows.extend(s.to_dict('records'))
# d. Outcome branches follow the complete modeling and fitting workflow.
title=env['title'];module=env['module'];arrow=env['arrow'];text=env['text']
title('d',190,'Clinical outcomes: unweighted 3-year / 5-year counts')
module(7,207,22,10,f'CN\nn = {counts["CN_MCI"]["3"]["all"]}')
module(48,199,39,11,f'No MCI or AD\n{counts["CN_MCI"]["3"]["event_free"]} / {counts["CN_MCI"]["5"]["event_free"]}')
module(48,215,39,11,f'First recorded MCI\n{counts["CN_MCI"]["3"]["target"]} / {counts["CN_MCI"]["5"]["target"]}')
arrow([(29,212),(37,212),(37,204.5),(47.5,204.5)])
arrow([(37,212),(37,220.5),(47.5,220.5)])
module(99,207,22,10,f'MCI\nn = {counts["MCI_AD"]["3"]["all"]}')
module(140,199,38,11,f'No AD dementia\n{counts["MCI_AD"]["3"]["event_free"]} / {counts["MCI_AD"]["5"]["event_free"]}')
module(140,215,38,11,f'AD dementia\n{counts["MCI_AD"]["3"]["target"]} / {counts["MCI_AD"]["5"]["target"]}')
arrow([(121,212),(129,212),(129,204.5),(139.5,204.5)])
arrow([(129,212),(129,220.5),(139.5,220.5)])
text(7,228.5,f'AD before MCI: {counts["CN_MCI"]["3"]["competing_ad"]} / {counts["CN_MCI"]["5"]["competing_ad"]}',7,'muted')
text(7,232.5,f'Incomplete follow-up: {counts["CN_MCI"]["3"]["unknown"]} / {counts["CN_MCI"]["5"]["unknown"]}',7,'muted')
text(99,228.5,f'Incomplete follow-up: {counts["MCI_AD"]["3"]["unknown"]} / {counts["MCI_AD"]["5"]["unknown"]}',7,'muted')
text(99,232.5,'No AD can include CN reversion.',7,'muted')
fig.canvas.draw();ren=fig.canvas.get_renderer()
bb=[(t,t.get_window_extent(ren)) for t in TEXT]
collisions=[[t.get_text(),u.get_text()] for i,(t,b) in enumerate(bb) for u,c in bb[i+1:] if min(b.x1,c.x1)-max(b.x0,c.x0)>.4 and min(b.y1,c.y1)-max(b.y0,c.y0)>.4]
outside=[t.get_text() for t,b in bb if not(fig.bbox.contains(b.x0,b.y0) and fig.bbox.contains(b.x1,b.y1))]
containment=[]
for i,(t,x,y,w,h) in enumerate(BOXES):
 b=t.get_window_extent(ren);params=next((v for v in panel_transforms.values() if i in v['module_indices']),None)
 if params:
  y=params['target']+params['scale']*(y-params['origin']);h*=params['scale']
 p0=ax.transData.transform((x,y+h));p1=ax.transData.transform((x+w,y))
 if b.x0<p0[0]+.3 or b.x1>p1[0]-.3 or b.y0<p0[1]+.3 or b.y1>p1[1]-.3:containment.append(t.get_text())
qc={'text_bbox_collisions':collisions,'out_of_canvas':outside,'module_text_clipping':containment,'size_mm':[183,250],'native_font_sizes_pt':sorted(set(t.get_fontsize() for t in TEXT))}
plt.rcParams.update({'savefig.bbox':None,'svg.fonttype':'none','svg.hashsalt':'prism-outcome-branches-20261019'})
