"""Native layout helpers retained from the current publication figures."""
from pathlib import Path
import os, importlib.util, json, hashlib, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE=Path(__file__).resolve().parent
STYLE=HERE/'style.py'
DATA=Path(os.environ['PRISM_AD_SOURCE_DATA'])
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,str(path))
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
sty=load('current_author_style',STYLE)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def export(*args):raise RuntimeError('Caller must capture the figure before exporting')

def geometry(fig):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();texts=list(fig.texts)
 for a in fig.axes:
  texts += [a.xaxis.label,a.yaxis.label]+list(a.texts)
  if a.get_legend():texts+=list(a.get_legend().get_texts())
  for aa,lim in [(a.xaxis,a.get_xlim()),(a.yaxis,a.get_ylim())]:
   for tick in aa.get_major_ticks():
    if min(lim)<=tick.get_loc()<=max(lim):texts += [tick.label1,tick.label2]
 for leg in fig.legends:texts+=list(leg.get_texts())
 texts=[t for t in texts if t.get_visible() and t.get_text().strip()]
 bb=[(t,t.get_window_extent(renderer)) for t in texts]
 return {'text_bbox_collisions':[[t.get_text(),u.get_text()] for i,(t,b) in enumerate(bb) for u,c in bb[i+1:] if min(b.x1,c.x1)-max(b.x0,c.x0)>.4 and min(b.y1,c.y1)-max(b.y0,c.y0)>.4],
 'out_of_canvas':[t.get_text() for t,b in bb if b.x0<-.2 or b.x1>fig.bbox.x1+.2 or b.y0<-.2 or b.y1>fig.bbox.y1+.2],
 'size_mm':list(np.round(fig.get_size_inches()*25.4,3))}

def signature(fig):
 # Data-space arrays: exclude figure-normalized layout artists.
 x=[]
 for a in fig.axes:
  x.append({'lines':[(np.asarray(v.get_xdata()).tolist(),np.asarray(v.get_ydata()).tolist()) for v in a.lines],
            'collections':[np.ma.asarray(v.get_offsets()).filled(np.nan).tolist() for v in a.collections]})
 return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=True).encode()).hexdigest()

def reflow(fig,oldH,newH,shift,drop):
 removed=[];sig=signature(fig)
 for t in list(fig.texts):
  if t.get_text() in drop:
   removed.append(t.get_text());t.remove();continue
  x,y=t.get_position();old_top=oldH*(1-y);t.set_position((x,1-shift(old_top)/newH))
  if t.get_fontweight()=='bold' and len(t.get_text())>1:t.set_fontweight('normal')
 for a in fig.axes:
  p=a.get_position();old_top=oldH*(1-p.y1)
  a.set_position([p.x0,1-(shift(old_top)+p.height*oldH)/newH,p.width,p.height*oldH/newH])
 for artist in list(fig.artists):
  if isinstance(artist,Rectangle):
   x,y=artist.get_xy();w,h=artist.get_width(),artist.get_height();top=oldH*(1-y-h)
   artist.set_y(1-(shift(top)+h*oldH)/newH);artist.set_height(h*oldH/newH)
 for leg in fig.legends:
  # Existing figure legend anchor is in figure coordinates.
  old=leg.get_bbox_to_anchor()._bbox
  x,y=old.x0,old.y0
  leg.set_bbox_to_anchor((x,1-shift(oldH*(1-y))/newH),transform=fig.transFigure)
 fig.set_size_inches(183/25.4,newH/25.4,forward=True)
 assert signature(fig)==sig
 return removed,sig

def setup(m):
 sty.apply_figure_style(font='Nimbus Sans',sizes=(8,7,6.5),grid=False)
 # Imported source modules keep their frozen data and plotting functions.
 m.STYLE=STYLE
 plt.rcParams['savefig.bbox']=None
 return m

def performance():
 m=setup(load('frozen_perf',HERE/'performance_base.py'))
 m.C['tab']='#9AA4AD';m.C['hgb']='#65717E'
 m.METHODS=[(model,name,m.C[code],marker) for (model,name,col,marker),code in zip(m.METHODS,['prism','lr','hgb','tab'])]
 m.make();fig=m.fig
 drop={'Prediction context: current clinical measures + PRS','Higher AP is better','Positive favors PRISM-AD','Added PRS information after cognition: matched PRISM-AD refits','Positive differences favor adding PRS (c–d)'}
 for l in list(fig.lines):l.set_visible(False)
 removed,sig=reflow(fig,180,163,lambda y:y-8 if y<102 else y-19,drop)
 fig.axes[0].set_xlabel('Average precision (higher is better)',fontsize=7)
 receipt={'native_script':str(HERE/'performance_base.py'),'native_script_sha256':sha(HERE/'performance_base.py'),'plotted_rows':m.ROWS,'significance_annotations':m.ANNOTATIONS}
 export(fig,3,receipt,removed,sig)

def overall():
 m=setup(load('frozen_overall',HERE/'prs_profiles_base.py'))
 receipt=json.loads((DATA/'plotting/PRS_quintile_summaries.json').read_text())
 m.save=lambda fig,name:None
 # summaries are the already frozen group/bin/fold rows, not newly fitted summaries
 m.signed_all_folds(receipt['signed_fold_summaries'],receipt['pooled_quintile_summaries'],5)
 fig=plt.gcf()
 drop={'Cognitive and genetic terms across inherited-risk groups','CN at prediction · 3-year first-MCI forecast · follow-up groups use all recorded visits','Open markers: bin n < 5; adjacent segments are thinner. Means remain separate by fitted fold.','Higher contributions raise predicted risk when all other model terms are fixed.'}
 removed,sig=reflow(fig,178,155,lambda y:y-15 if y<105 else y-20,drop)
 for t in fig.texts:t.set_text(re.sub(r' \(n = \d+\)$','',t.get_text()))
 from matplotlib.lines import Line2D
 fig.legends[-1].remove()
 fold_handles=[Line2D([],[],color=m.C['muted'],lw=1,ls=ls,label='Fold '+f) for f,ls in [('1','-'),('2','--'),('3',':')]]
 fold_handles.append(Line2D([],[],color=m.C['muted'],marker='o',mfc='white',ms=3,ls='None',label='n < 5'))
 fig.legend(handles=fold_handles,loc='center',bbox_to_anchor=(.53,1-15/155),ncol=4,frameon=False,fontsize=7,handlelength=2,columnspacing=1.4)
 export(fig,4,{'native_receipt_path':str(DATA/'plotting/PRS_quintile_summaries.json'),'native_receipt_sha256':sha(DATA/'plotting/PRS_quintile_summaries.json'),'signed_fold_summaries':receipt['signed_fold_summaries']},removed,sig)

def contrasts():
 m=setup(load('frozen_contrasts',HERE/'contrasts_base.py'))
 # Capture the exact figure at the native audit boundary; skip historical export/QC bookkeeping.
 class Captured(Exception):pass
 captured={}
 def capture(fig,*args):
  captured['fig']=fig;raise Captured()
 original=m.old.audit_figure;m.old.audit_figure=capture
 try:m.main()
 except Captured:pass
 finally:m.old.audit_figure=original
 fig=captured['fig']
 drop={'PRISM-AD components in CN-to-MCI forecasts','Negative = lower in high-PRS last-CN; positive = higher','How clinical measurements become risk contributions','Within a forecast: higher contribution → higher predicted risk, other terms fixed'}
 removed,sig=reflow(fig,210,190,lambda y:y-4 if y<75 else y-16,drop)
 export(fig,5,{'native_script_sha256':sha(HERE/'contrasts_base.py'),'data_provenance':m.provenance,'significance_annotations':m.annotations},removed,sig)
