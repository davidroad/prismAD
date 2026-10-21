#!/usr/bin/env python3
"""Draw loss diagnostics only after all original-model replay gates pass."""
from pathlib import Path
import importlib.util,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

import os
HERE=Path(__file__).resolve().parent
D=Path(os.environ['PRISM_AD_SOURCE_DATA'])/'optimization'
STYLE=HERE/'style.py'
spec=importlib.util.spec_from_file_location('author_figure_style',STYLE)
style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
style.apply_figure_style(font='DejaVu Sans',sizes=(9,8,7),grid=False,frame='open')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build():
 gate=json.loads((D/'replay_verification.json').read_text())
 assert gate['status']=='PASS' and len(gate['run_audits'])==24
 for a in gate['run_audits']:
  assert a['all_60_original_training_losses_exact'] and a['all_final_state_tensors_exact'] and a['all_final_validation_probabilities_exact']
 data=pd.read_csv(D/'learning_curves.csv')
 expected={(t,h,m,f) for t in ('CN_MCI','MCI_AD') for h in (3,5) for m in ('PRISM_current','PRISM_history') for f in (1,2,3)}
 assert set(data[['task','horizon','model','fold']].itertuples(index=False,name=None))==expected
 assert len(data)==1440 and not data.duplicated(['task','horizon','model','fold','epoch']).any()
 assert np.isfinite(data[['training_BCE','validation_IPCW_BCE','original_online_training_BCE']].to_numpy()).all()
 assert (data[['training_BCE','validation_IPCW_BCE']]>=0).all().all()
 for key,part in data.groupby(['task','horizon','model','fold']):
  assert part.epoch.tolist()==list(range(1,61)),key
 palette={'training_BCE':'#0072B2','validation_IPCW_BCE':'#D55E00'}
 widths=240/25.4;heights=139/25.4
 fig,axes=plt.subplots(2,4,figsize=(widths,heights),sharex=True,sharey='col')
 fig.subplots_adjust(left=.075,right=.984,bottom=.15,top=.867,wspace=.38,hspace=.61)
 models=['PRISM_current','PRISM_history'];columns=[('CN_MCI',3),('CN_MCI',5),('MCI_AD',3),('MCI_AD',5)]
 panels=[];summaries=[]
 for r,model in enumerate(models):
  for c,(task,horizon) in enumerate(columns):
   ax=axes[r,c];part=data[(data.model==model)&(data.task==task)&(data.horizon==horizon)]
   letter=chr(97+r*4+c)
   for metric,color in palette.items():
    ls='-' if metric=='training_BCE' else '--'
    for fold,p in part.groupby('fold'):
     ax.plot(p.epoch,p[metric],color=color,lw=.65,alpha=.32,ls=ls,zorder=1)
    mean=part.groupby('epoch',sort=True)[metric].mean()
    ax.plot(mean.index,mean.values,color=color,lw=1.7,ls=ls,zorder=3)
   same_column=data[(data.task==task)&(data.horizon==horizon)]
   lo=float(same_column[['training_BCE','validation_IPCW_BCE']].min().min())
   hi=float(same_column[['training_BCE','validation_IPCW_BCE']].max().max())
   pad=.07*(hi-lo)
   ax.set_ylim(max(0,lo-pad),hi+pad)
   ax.set_xlim(1,60);ax.set_xticks([1,20,40,60]);ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
   ax.tick_params(labelleft=True,length=2.7,width=.7,pad=2)
   ax.set_title(('CN → MCI' if task=='CN_MCI' else 'MCI → AD')+f', {horizon} years',pad=7,fontweight='normal')
   if c==0:ax.set_ylabel('Binary cross-entropy',labelpad=5)
   if r==1:ax.set_xlabel('Epoch',labelpad=4)
   pos=ax.get_position()
   fig.text(pos.x0-.026,pos.y1+.023,letter,fontweight='bold',fontsize=11,ha='left',va='baseline')
   panels.append(dict(letter=letter,task=task,horizon=horizon,model=model,
    source='learning_curves.csv',source_sha256=sha(D/'learning_curves.csv'),n_fold_curves=3,epochs=[1,60],
    y_limits=list(ax.get_ylim()),mean_curve='unweighted across folds',line_types='thin individual folds; thick mean',
    training_definition='unpenalized mean BCE on fixed original IPCW resample',
    validation_definition='unpenalized capped-IPCW BCE / sum weights; all known eligible held-out visits'))
   vm=part.groupby('epoch').validation_IPCW_BCE.mean()
   summaries.append(dict(panel=letter,task=task,horizon=horizon,model=model,
    mean_validation_BCE_epoch1=float(vm.iloc[0]),mean_validation_BCE_epoch60=float(vm.iloc[-1]),
    minimum_mean_validation_BCE=float(vm.min()),epoch_of_minimum=int(vm.idxmin()),
    final_minus_minimum=float(vm.iloc[-1]-vm.min()),descriptive_only=True,not_used_for_checkpoint_selection=True))
 fig.text(.53,.962,'PRISM-AD using current assessments',ha='center',va='center',fontsize=9)
 fig.text(.53,.518,'PRISM-AD incorporating prior assessments',ha='center',va='center',fontsize=9)
 handles=[Line2D([],[],color=palette['training_BCE'],lw=1.7,label='Training'),
  Line2D([],[],color=palette['validation_IPCW_BCE'],lw=1.7,ls='--',label='Validation')]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.012),ncol=2,frameon=False,handlelength=2.3,columnspacing=2.3)
 fig.text(.984,.041,'Lower loss is better',ha='right',va='center',fontsize=8,color='#444444')
 fig.canvas.draw()
 # Locators retain extra tick objects outside the view; Matplotlib does not draw those.
 # Hide their unused labels before testing the actual rendered text geometry.
 for ax in axes.flat:
  low,high=ax.get_ylim()
  for tick in ax.yaxis.get_major_ticks():
   if not low-1e-12<=tick.get_loc()<=high+1e-12:
    tick.label1.set_visible(False);tick.label2.set_visible(False)
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
 collisions=[]
 for i,(a,ba) in enumerate(texts):
  for b,bb in texts[i+1:]:
   if ba.overlaps(bb):collisions.append([a.get_text(),b.get_text()])
 outside=[t.get_text() for t,bb in texts if bb.x0<-1 or bb.y0<-1 or bb.x1>fig.bbox.width+1 or bb.y1>fig.bbox.height+1]
 assert not collisions,collisions
 assert not outside,outside
 return fig
