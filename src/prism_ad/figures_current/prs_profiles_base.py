#!/usr/bin/env python3
"""Compare observed relative-term displays using one frozen CN-at-prediction model.
No prognostic fitting, inference tests, individual IDs, or individual CSV exports.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import os
HERE=Path(__file__).resolve().parent
STYLE=HERE/'style.py'
sp=importlib.util.spec_from_file_location('author_style',STYLE)
sty=importlib.util.module_from_spec(sp);sp.loader.exec_module(sty)
sty.apply_figure_style(font='Nimbus Sans',sizes=(8,7,6.5),grid=False)
plt.rcParams.update({'savefig.bbox':None,'svg.fonttype':'none','svg.hashsalt':'prism-relative-components-20261017','pdf.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white','text.color':'#24313B','axes.labelcolor':'#24313B','axes.edgecolor':'#65717E','xtick.color':'#65717E','ytick.color':'#24313B','axes.labelpad':4,'xtick.major.pad':2,'ytick.major.pad':2})
C=dict(ink='#24313B',muted='#65717E',cognition='#D8892B',prs='#7656A4',prism='#007F79',line='#D2DADD')
GROUPS=[('All','All participants'),('CN','Last classified CN'),('MCI','Last classified MCI'),('AD','Later recorded AD')]
def save(fig,name):pass

def txt(fig,x,y,s,sz=8,ha='left',col='ink',rotation=0,weight='normal'):
    w,h=fig.get_size_inches()*25.4
    return fig.text(x/w,1-y/h,s,fontsize=sz,ha=ha,va='center',color=C.get(col,col),rotation=rotation,fontweight=weight)

def axis(fig,x,y,w,h):
    fw,fh=fig.get_size_inches()*25.4
    a=fig.add_axes([x/fw,1-(y+h)/fh,w/fw,h/fh]);a.spines['top'].set_visible(False);a.spines['right'].set_visible(False);return a

def canvas(w,h):return plt.figure(figsize=(w/25.4,h/25.4),dpi=600)

def row(rows,group,b):return next(r for r in rows if r['group']==group and r['bin']==b)

def signed_all_folds(rows_by_fold, pooled, nbins):
    fig=canvas(183,178)
    txt(fig,5,7,'Cognitive and genetic terms across inherited-risk groups')
    txt(fig,5,15,'CN at prediction · 3-year first-MCI forecast · follow-up groups use all recorded visits',7,col='muted')
    colors=[Line2D([],[],color=C['cognition'],marker='o',lw=1,ms=3,label='Cognitive-test component'),Line2D([],[],color=C['prs'],marker='s',lw=1,ms=3,label='PRS term')]
    folds=[Line2D([],[],color=C['muted'],lw=1,ls=ls,label='Fold '+f) for f,ls in [('1','-'),('2','--'),('3',':')]]
    fig.legend(handles=colors,loc='center',bbox_to_anchor=(.53,1-23/178),ncol=2,frameon=False,fontsize=7,handlelength=1.5,columnspacing=2)
    fig.legend(handles=folds,loc='center',bbox_to_anchor=(.53,1-31/178),ncol=3,frameon=False,fontsize=7,handlelength=2,columnspacing=2)
    vals=[r[k] for rows in rows_by_fold.values() for r in rows for k in ['mean_signed_C','mean_signed_G'] if r[k] is not None]
    lo,hi=min(vals+[0]),max(vals+[0]);pad=(hi-lo)*.08
    styles={'1':'-','2':'--','3':':'}
    for i,(key,name) in enumerate(GROUPS):
        x=25+(i%2)*88;y=49+(i//2)*65
        a=axis(fig,x,y,63,38)
        total=sum(row(pooled,key,b)['n'] for b in range(1,nbins+1))
        txt(fig,x+31.5,y-9,('Overall' if key=='All' else name)+' (n = '+str(total)+')',8,ha='center')
        txt(fig,8+(i%2)*88,y-9,'abcd'[i],10,weight='bold')
        for f,rows in rows_by_fold.items():
            sub=[row(rows,key,b) for b in range(1,nbins+1)]
            for k,col,m in [('mean_signed_C',C['cognition'],'o'),('mean_signed_G',C['prs'],'s')]:
                z=[r[k] if r[k] is not None else np.nan for r in sub]
                # One curve per fitted fold. Thin segments flag adjacent sparse cells.
                for j in range(nbins-1):
                    if np.isfinite(z[j]) and np.isfinite(z[j+1]):
                        low=min(sub[j]['n'],sub[j+1]['n'])<5
                        a.plot([j+1,j+2],[z[j],z[j+1]],color=col,ls=styles[f],lw=.6 if low else 1,alpha=.65 if low else .95,zorder=2)
                for j,r in enumerate(sub):
                    if z[j] is not None and np.isfinite(z[j]):
                        a.plot(j+1,z[j],marker=m,ls='None',ms=3.2,mec=col,mew=.7,mfc='white' if r['n']<5 else col,zorder=3)
        a.axhline(0,color=C['muted'],lw=.6,ls=':',zorder=0)
        a.set(xlim=(.6,nbins+.4),ylim=(lo-pad,hi+pad),xticks=range(1,nbins+1),xticklabels=['Q'+str(b) for b in range(1,nbins+1)])
        a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        if i%2:a.tick_params(labelleft=False)
    txt(fig,5,98,'Mean signed contribution (log-odds)',8,ha='center',rotation=90)
    txt(fig,100,160,'Shared raw-PRS quintile',8,ha='center')
    txt(fig,5,168,'Open markers: bin n < 5; adjacent segments are thinner. Means remain separate by fitted fold.',6.5,col='muted')
    txt(fig,5,174,'Higher contributions raise predicted risk when all other model terms are fixed.',6.5,col='muted')
    save(fig,'Signed_terms_all_folds_quintiles')
