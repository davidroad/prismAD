#!/usr/bin/env python3
"""Reader-oriented Figure 3; redraw frozen estimates, never fit or test."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys
HERE=Path(__file__).resolve().parent
OUT=Path(os.environ['PRISM_AD_RENDER_WORK'])
SOURCE=Path(os.environ['PRISM_AD_SOURCE_DATA'])/'plotting'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from prism_ad import plot_prognosis as old
STYLE=HERE/'style.py'
spec=importlib.util.spec_from_file_location('author_figure_style',str(STYLE))
style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
style.apply_figure_style(font='Nimbus Sans',sizes=(8,7,6.5),grid=False)
C=dict(ink='#24313B',muted='#65717E',prism='#007F79',cognition='#D8892B',prs='#7656A4',ad='#C75058',other_CN='#5792BB',HGB='#7D8995',TabPFN='#CB6040')
plt.rcParams.update({'savefig.bbox':None,'svg.fonttype':'none','svg.hashsalt':'prism-ad-reader-logic-figure3-20261017','pdf.fonttype':42,'text.color':C['ink'],'axes.labelcolor':C['ink'],'axes.edgecolor':C['muted'],'xtick.color':C['ink'],'ytick.color':C['ink'],'axes.titlelocation':'left'})
W,H=183,210
annotations=[];provenance=[];letters=[];axes=[]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def star(q):return '***' if q<.001 else '**' if q<.01 else '*' if q<.05 else ''
def text(fig,x,y,value,size=8,**kwargs):return fig.text(x/W,y/H,value,fontsize=size,**kwargs)
def axis(fig,x,y,w,h):
    ax=fig.add_axes([x/W,y/H,w/W,h/H]);axes.append(ax);old.clean(ax)
    ax.tick_params(direction='out',length=2.5,width=.6,pad=2);return ax
def panel_title(fig,panel,title,xcenter,y,primary=False):
    letters.append(text(fig,2 if xcenter<W/2 else 96,y,panel,size=10,weight='bold',ha='left',va='top'))
    return text(fig,xcenter,y,title,ha='center',va='top',weight='bold' if primary else 'normal',linespacing=1.25)
def horizon_key(ax,where=(.5,1.115)):
    handles=[Line2D([],[],marker=m,ls='None',color=C['ink'],ms=3.7,mfc=C['ink'] if h==3 else 'white',mew=.8,label=f'{h}-year') for h,m in [(3,'o'),(5,'s')]]
    return ax.legend(handles=handles,loc='center',bbox_to_anchor=where,ncol=2,frameon=False,handlelength=1.2,handletextpad=.4,columnspacing=1.3,borderaxespad=0,fontsize=7)
def group_contrast(fig,panel,right,x):
    ax=axis(fig,x,147,50,40)
    components=[('cognition_main','Cognitive-test\nrisk component',C['cognition'],0),('PRS','PRS term',C['prs'],1),('education_main','Education term',C['muted'],2),('risk_logit','Total predicted\nlog odds',C['ink'],3.45)]
    plotted=[]
    for component,label,color,y in components:
        for h,marker,dy in [(3,'o',-.16),(5,'s',.16)]:
            r=old.one(old.contrasts,model='PRISM_current',scope='primary_observed',right=right,component=component,horizon=h);assert r.status=='ok'
            ax.errorbar(r.beta,y+dy,xerr=[[r.beta-r.ci_low],[r.ci_high-r.beta]],fmt=marker,ms=3.7,mfc=color if h==3 else 'white',mew=.8,color=color,lw=.85,capsize=2,zorder=3)
            label_star=star(r.q_all_128) or ('(*)' if r.p<.05 else '')
            if label_star:ax.text(r.ci_high+.045,y+dy,label_star,fontsize=7,weight='bold',ha='left',va='center',color=C['ink'])
            item=dict(figure=3,panel=panel,component=component,right=right,horizon=h,estimate=float(r.beta),ci_low=float(r.ci_low),ci_high=float(r.ci_high),p=float(r.p),q=float(r.q_all_128),label=label_star,criterion='BH q',family='128 contribution contrasts',n_high_PRS_last_CN=int(r.n_resilience_CN),n_other_CN=int(r.n_other_CN_no_AD),n_CN_later_AD=int(r.n_CN_to_AD))
            annotations.append(item);plotted.append(item)
    ax.axvline(0,color=C['muted'],ls=':',lw=.65,zorder=1);ax.axhline(2.76,color='#D9DEE3',lw=.6,zorder=1)
    ax.set(yticks=[z[3] for z in components],yticklabels=[z[1] for z in components],ylim=(4.03,-.55),xlim=(-.8,.9),xticks=[-.5,0,.5],xlabel='Adjusted difference (log odds)')
    ax.tick_params(axis='y',length=0);horizon_key(ax)
    provenance.append(dict(panel=panel,old_panel='f' if right=='CN_to_AD' else 'e',source='adjusted_component_contrasts.csv',source_row_filter={'model':'PRISM_current','scope':'primary_observed','right':right,'components':[z[0] for z in components],'horizons':[3,5]},plotted_estimates=8,plotted_CIs=8,plotted_starred_estimates=sum(bool(x['label']) for x in plotted),row_values=plotted))
    return ax
def curves(fig,panel,feature,left,bottom,color):
    d=old.curves[(old.curves.model=='PRISM_current')&(old.curves.feature==feature)]
    ref=float(d.reference_raw_value.iloc[0]);assert np.allclose(d.reference_raw_value,ref)
    low,high=float(d.centered_main_logit.min()),float(d.centered_main_logit.max());pad=.13*(high-low)
    panel_axes=[];curve_records=[]
    for h,offset in [(3,19),(5,0)]:
        ax=axis(fig,left,bottom+offset,48.5,12.5)
        for fold,ls,marker in [(1,'-','o'),(2,'--','s'),(3,':','^')]:
            z=d[(d.horizon==h)&(d.fold==fold)].sort_values('raw_value');assert len(z)==41
            ax.plot(z.raw_value,z.centered_main_logit,color=color,ls=ls,lw=.85,marker=marker,markevery=[5,15,25],ms=2.6,mfc='white',mew=.65)
            curve_records.append(dict(horizon=h,fold=fold,n_vertices=len(z),points_sha256=hashlib.sha256(z[['raw_value','centered_main_logit']].to_numpy().tobytes()).hexdigest()))
        ax.axhline(0,color=C['muted'],lw=.55,ls=':');ax.axvline(ref,color=C['muted'],lw=.55,ls=':')
        ax.set_ylim(low-pad,high+pad);ax.margins(x=.03)
        ax.yaxis.set_major_locator(MaxNLocator(3,prune='both'));ax.xaxis.set_major_locator(MaxNLocator(4,prune='both'))
        ax.text(1.025,.5,f'{h} y',transform=ax.transAxes,fontsize=7,ha='left',va='center')
        if h==3:ax.tick_params(axis='x',labelbottom=False)
        panel_axes.append(ax)
    labels={'PHC_MEM_z':'Memory test score (PHC)','global_z':'Mean cognitive score (PHC)','PHC_Education':'Education (years)'}
    panel_axes[-1].set_xlabel(labels[feature])
    text(fig,left-13.5,bottom+15.75,'Risk contribution (log odds)',rotation=90,ha='center',va='center')
    handles=[Line2D([],[],color=color,ls=ls,marker=marker,ms=2.6,mfc='white',lw=.85,label=f'Fold {fold}') for fold,ls,marker in [(1,'-','o'),(2,'--','s'),(3,':','^')]]
    panel_axes[0].legend(handles=handles,loc='center',bbox_to_anchor=(.5,1.37),ncol=3,frameon=False,handlelength=1.5,columnspacing=.8,handletextpad=.35,borderaxespad=0,fontsize=7)
    provenance.append(dict(panel=panel,old_panel={'c':'a','d':'b','e':'c'}[panel],source='training_supported_shape_curves.csv',source_row_filter={'model':'PRISM_current','feature':feature,'horizons':[3,5],'folds':[1,2,3]},reference_raw_value=ref,centered_at_reference=True,plotted_curves=6,plotted_vertices=246,curves=curve_records,scientific_note='Horizontal axis is the recorded measurement scale; vertical axis is a fitted centered prediction term.'))
    return panel_axes
def high_low(fig):
    ax=axis(fig,129,19,50,31.5);records=[]
    labels=['Age','Education','Mean cognitive\nscore','Memory','Executive','Language','Visuospatial']
    for i,(feature,unused) in enumerate(old.FEATURES):
        color=C['muted'] if i<2 else C['cognition']
        for h,marker,dy in [(3,'o',-.13),(5,'s',.13)]:
            r=old.one(old.stability,model='PRISM_current',horizon=h,feature=feature)
            vals=np.array([r['delta_fold'+str(f)] for f in (1,2,3)]);yy=i+dy+np.array([-.04,0,.04])
            ax.scatter(vals,yy,s=13,marker=marker,facecolor=color if h==3 else 'white',edgecolor=color,lw=.7,zorder=3)
            records.append(dict(feature=feature,horizon=h,raw_low=float(r.raw_low),raw_high=float(r.raw_high),individual_fold_differences=vals.tolist(),individual_y_offsets=yy.tolist()))
    ax.axvline(0,color=C['muted'],ls=':',lw=.65)
    ax.set(yticks=range(7),yticklabels=labels,ylim=(6.6,-.6),xlim=(-1.6,1.6),xticks=[-1.5,0,1.5],xlabel='High − low contribution (log odds)')
    ax.xaxis.label.set_fontsize(7);ax.tick_params(axis='y',length=0);horizon_key(ax,where=(.5,1.14))
    provenance.append(dict(panel='f',old_panel='d',source='shape_stability.csv',source_row_filter={'model':'PRISM_current','features':[z[0] for z in old.FEATURES],'horizons':[3,5]},plotted_individual_fit_points=42,intervals_shown=False,row_values=records))
    return ax
def main():
    previous_path=OUT/'figure3_receipt.json'
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else {}
    OUT.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=180)
    text(fig,W/2,207.5,'PRISM-AD components in CN-to-MCI forecasts',ha='center',va='top')
    panel_title(fig,'a','High-PRS last-CN −\nCN later with AD',59,199,primary=True)
    panel_title(fig,'b','High-PRS last-CN −\nother CN',153,199)
    group_contrast(fig,'a','CN_to_AD',34);group_contrast(fig,'b','other_CN_no_AD',129)
    text(fig,W/2,133,'Negative = lower in high-PRS last-CN; positive = higher',size=7,ha='center',va='center')
    text(fig,W/2,126.5,'How clinical measurements become risk contributions',ha='center',va='center')
    text(fig,W/2,122.5,'Within a forecast: higher contribution → higher predicted risk, other terms fixed',size=7,ha='center',va='center')
    panel_title(fig,'c','Memory function',59,116);panel_title(fig,'d','Mean cognitive score function',153,116)
    curves(fig,'c','PHC_MEM_z',34,75,C['cognition']);curves(fig,'d','global_z',129,75,C['cognition'])
    panel_title(fig,'e','Education function',59,60);panel_title(fig,'f','Function changes across folds',153,60)
    curves(fig,'e','PHC_Education',34,19,C['muted']);high_low(fig)
    return old.audit_figure(fig,axes,letters)
