#!/usr/bin/env python3
"""Redesigned Figure 2: prediction context and matched PRS increments, frozen rows only."""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import os
HERE=Path(__file__).resolve().parent
OUT=Path(os.environ['PRISM_AD_RENDER_WORK'])
SOURCE=Path(os.environ['PRISM_AD_SOURCE_DATA'])/'plotting'
STYLE=HERE/'style.py'
sp=importlib.util.spec_from_file_location('author_style',STYLE)
sty=importlib.util.module_from_spec(sp);sp.loader.exec_module(sty)
sty.apply_figure_style(font='Nimbus Sans',sizes=(8,7,6.5),grid=False)
plt.rcParams.update({'savefig.bbox':None,'svg.fonttype':'none','svg.hashsalt':'prism-reader-figure2-20261017','pdf.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white','text.color':'#24313B','axes.labelcolor':'#24313B','xtick.color':'#65717E','ytick.color':'#24313B','axes.edgecolor':'#65717E','axes.labelpad':4,'xtick.major.pad':2,'ytick.major.pad':3})
W,H=183,180
C=dict(prism='#007F79',lr='#24313B',hgb='#7D8995',tab='#CB6040',ink='#24313B',muted='#65717E',line='#D2DADD')
METHODS=[('PRISM_current','PRISM-AD',C['prism'],'o'),('LR_current','Logistic',C['lr'],'D'),('HGB_current','HGB',C['hgb'],'s'),('TabPFN_current','TabPFN',C['tab'],'^')]
ENDS=[('CN_MCI',3),('CN_MCI',5),('MCI_AD',3),('MCI_AD',5)]
ENDLABELS=['CN → first MCI, 3 y','CN → first MCI, 5 y','MCI → AD, 3 y','MCI → AD, 5 y']
FILENAMES=['first_visit_metrics_recomputed.csv','current_model_paired_comparisons.csv','paired_AP_tests.csv','genetic_contrasts.csv']
metrics,paired,tests,genetic=[pd.read_csv(SOURCE/f) for f in FILENAMES]
ROWS,ANNOTATIONS=[],[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def one(df,**filters):
    d=df
    for k,v in filters.items():d=d[d[k].eq(v)]
    assert len(d)==1,(filters,len(d))
    return d.iloc[0]
def native(v):
    if isinstance(v,(np.integer,np.floating,np.bool_)):return v.item()
    return v
def record(panel,filename,row,keys):
    ROWS.append(dict(panel=panel,source_csv=filename,row_key={k:native(row[k]) for k in keys},row={k:native(v) for k,v in row.items()}))
def txt(x,y,s,sz=8,ha='left',color='ink',weight='normal'):
    return fig.text(x/W,1-y/H,s,fontsize=sz,ha=ha,va='center',color=C.get(color,color),weight=weight)
def axis(x,y,w,h):
    a=fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
    a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
    return a
def common(a):
    a.set(yticks=range(4),yticklabels=ENDLABELS,ylim=(3.55,-.55))
    a.tick_params(axis='y',length=0)
    a.axhline(1.5,color=C['line'],lw=.5,zorder=0)
    return a
def header(letter,title,x,y):
    txt(x,y,letter,10,weight='bold');txt(x+6,y,title)
def legend(a,methods,ncol):
    handles=[Line2D([],[],color=c,marker=m,ls='None',ms=4,markeredgewidth=.45,markeredgecolor='white',label=name) for _,name,c,m in methods]
    a.legend(handles=handles,loc='lower center',bbox_to_anchor=(.46,1.09),ncol=ncol,frameon=False,fontsize=7,handlelength=.7,handletextpad=.35,columnspacing=.9,borderaxespad=0)
def make():
    global fig
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=600)
    txt(5,6,'Prediction context: current clinical measures + PRS')
    header('a','Average precision by clinical stage',5,15)
    header('b','Paired prediction differences',96,15)
    a=common(axis(29,36,56,43))
    b=common(axis(122,36,55,43))
    for j,(model,name,col,marker) in enumerate(METHODS):
        for i,(task,h) in enumerate(ENDS):
            r=one(metrics,task=task,horizon=h,model=model)
            a.scatter(r.auprc,i+(j-1.5)*.16,c=col,marker=marker,s=18,zorder=4,edgecolors='white',linewidths=.35)
            record('a',FILENAMES[0],r,['task','horizon','model'])
    for i,(task,h) in enumerate(ENDS):
        r=one(metrics,task=task,horizon=h,model='PRISM_current')
        a.plot([r.weighted_prevalence]*2,[i-.37,i+.37],color=C['muted'],lw=.7,ls=':',zorder=1)
    a.set(xlim=(0,1),xticks=[0,.25,.5,.75,1],xlabel='Average precision (AP)')
    legend(a,METHODS,4)
    for j,(reference,name,col,marker) in enumerate(METHODS[1:]):
        for i,(task,h) in enumerate(ENDS):
            r=one(paired,task=task,horizon=h,model='PRISM_current',reference=reference,metric='auprc')
            t=one(tests,task=task,horizon=h,subset='first_eligible_landmark',model='PRISM_current',reference=reference)
            assert abs(r.difference-t.difference)<1e-12
            assert abs(r.ci_low-t.ci_low_original500)<1e-12 and abs(r.ci_high-t.ci_high_original500)<1e-12
            assert t.family_size==28 and t.bootstrap_replicates==9999 and r.n_bootstrap==500
            y=i+(j-1)*.28
            b.errorbar(r.difference,y,xerr=[[r.difference-r.ci_low],[r.ci_high-r.difference]],fmt=marker,color=col,ms=3.7,mfc=col,mew=.65,lw=.8,capsize=1.8,zorder=3)
            star='***' if t.p<.001 else '**' if t.p<.01 else '*' if t.p<.05 else ''
            if star:b.text(.151,y,star,ha='center',va='center',fontsize=7,color=C['ink'],weight='bold')
            ANNOTATIONS.append(dict(figure=2,panel='b',task=task,horizon=h,model='PRISM_current',reference=reference,p=float(t.p),q=float(t.q),label=star,criterion='nominal two-sided P',family='28 first-visit AP tests; Supplementary Table 4',interval_source='original 500-resample participant-bootstrap percentile CI',p_source='9999-resample null-centered paired participant bootstrap'))
            record('b',FILENAMES[1],r,['task','horizon','model','reference','metric'])
    b.axvline(0,color=C['muted'],ls=':',lw=.7,zorder=0)
    b.set(xlim=(-.15,.17),xticks=[-.1,0,.1],xlabel='AP difference\n(PRISM-AD − comparator)')
    legend(b,METHODS[1:],3)
    txt(29,95,'Higher AP is better',7,color='muted')
    txt(122,95,'Positive favors PRISM-AD',7,color='muted')
    fig.lines.append(Line2D([5/W,178/W],[1-102/H]*2,transform=fig.transFigure,color=C['line'],lw=.65))
    txt(5,108,'Added PRS information after cognition: matched PRISM-AD refits')
    header('c','PRS increment in AUROC',5,117)
    header('d','PRS increment in AP',96,117)
    c=common(axis(29,129,56,35))
    d=common(axis(122,129,55,35))
    for panel,a,metric in [('c',c,'weighted_AUROC'),('d',d,'weighted_AP')]:
        for i,(task,h) in enumerate(ENDS):
            r=one(genetic,task=task,horizon=h,model='PRISM',fold='pooled_123',subset='first_eligible_landmark',metric=metric)
            assert r.contrast=='plusPRS_minus_noPRS' and r.n_bootstrap==500
            if metric=='weighted_AP':
                ap=one(metrics,task=task,horizon=h,model='PRISM_current').auprc
                assert abs(ap-r.plusPRS)<1e-12
            a.errorbar(r.difference,i,xerr=[[r.difference-r.ci_low],[r.ci_high-r.difference]],fmt='o',color=C['prism'],ms=4,mfc=C['prism'],mew=.65,lw=.9,capsize=2,zorder=3)
            record(panel,FILENAMES[3],r,['task','horizon','fold','model','subset','metric','contrast'])
        a.axvline(0,color=C['muted'],ls=':',lw=.7,zorder=0)
    c.set(xlim=(-.02,.04),xticks=[-.02,0,.02,.04],xlabel='AUROC difference (+PRS − no PRS)')
    d.set(xlim=(-.04,.04),xticks=[-.04,-.02,0,.02,.04],xlabel='AP difference (+PRS − no PRS)')
    txt(92,177,'Positive differences favor adding PRS (c–d)',7,ha='center',color='muted')
    assert len(ROWS)==36 and sum(r['panel']=='a' for r in ROWS)==16 and sum(r['panel']=='b' for r in ROWS)==12
    return [fig.axes[0],b,c,d]
def geometry(fig):
    fig.canvas.draw();r=fig.canvas.get_renderer();texts=list(fig.texts)
    for a in fig.axes:
        texts.extend([a.xaxis.label,a.yaxis.label]+list(a.texts))
        for axis,lim in [(a.xaxis,a.get_xlim()),(a.yaxis,a.get_ylim())]:
            for tick in axis.get_major_ticks():
                if min(lim)<=tick.get_loc()<=max(lim):texts.extend([tick.label1,tick.label2])
        if a.get_legend():texts.extend(a.get_legend().get_texts())
    texts=[t for t in texts if t.get_visible() and t.get_text().strip()]
    items=[(t,t.get_window_extent(r)) for t in texts]
    overlaps=[[t.get_text(),u.get_text()] for i,(t,b) in enumerate(items) for u,c in items[i+1:] if min(b.x1,c.x1)-max(b.x0,c.x0)>.4 and min(b.y1,c.y1)-max(b.y0,c.y0)>.4]
    outside=[t.get_text() for t,b in items if b.x0<fig.bbox.x0-.2 or b.x1>fig.bbox.x1+.2 or b.y0<fig.bbox.y0-.2 or b.y1>fig.bbox.y1+.2]
    legend_hits=[]
    for a in fig.axes:
        if a.get_legend():
            lb=a.get_legend().get_window_extent(r)
            if any(lb.overlaps(b.get_window_extent(r)) for b in fig.axes):legend_hits.append('legend intersects data rectangle')
    return dict(size_mm=[W,H],font_sizes_pt=sorted(set(t.get_fontsize() for t in texts)),text_bbox_collisions=overlaps,out_of_canvas=outside,legend_data_overlaps=legend_hits,plotted_data_rows=36)
