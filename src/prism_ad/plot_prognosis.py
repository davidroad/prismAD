from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from .figure_style import apply_figure_style
from .runtime import source_data
DATA=source_data()/'plotting'


apply_figure_style(font='Nimbus Sans', sizes=(7,6.5,6.5), grid=False)

C = dict(prism='#087F78',hgb='#68727D',tab='#C45A36',lr='#303B46',
         cognition='#D8892B',prs='#7656A4',cn='#5B8DB8',ad='#B44848',
         ink='#202124',grey='#59636E',grid='#D9DEE3')

plt.rcParams.update({'font.family':'Nimbus Sans','font.size':7,
    'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
    'legend.fontsize':6.5,'axes.linewidth':.6,'lines.linewidth':.85,
    'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,
    'ytick.major.size':2.5,'xtick.major.pad':2,'ytick.major.pad':2,
    'axes.labelpad':3,'axes.edgecolor':C['grey'],'text.color':C['ink'],
    'axes.labelcolor':C['ink'],'xtick.color':C['ink'],'ytick.color':C['ink'],
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'savefig.facecolor':'white','figure.facecolor':'white','axes.facecolor':'white',
    'savefig.bbox':None,'svg.hashsalt':'prism-ad-current-evidence'})


ENDPOINTS=[('CN_MCI',3),('CN_MCI',5),('MCI_AD',3),('MCI_AD',5)]

ENDLABELS=['CN → MCI, 3 y','CN → MCI, 5 y','MCI → AD, 3 y','MCI → AD, 5 y']

METHODS=[('PRISM_current','PRISM-AD',C['prism'],'o'),('LR_current','Logistic',C['lr'],'D'),
         ('HGB_current','HGB',C['hgb'],'s'),('TabPFN_current','TabPFN',C['tab'],'^')]

FEATURES=[('visit_age','Age'),('PHC_Education','Education'),('global_z','Global cognition'),
          ('PHC_MEM_z','Memory'),('PHC_EXF_z','Executive'),('PHC_LAN_z','Language'),('PHC_VSP_z','Visuospatial')]

GROUPS=[('resilience_CN','GRPa-PRS CN',C['prs'],'-'),
        ('other_CN_no_AD','Other CN',C['cn'],'--'),('CN_to_AD','CN → AD',C['ad'],':')]

COMPONENTS=[('PRS','PRS',C['prs']),('cognition_main','Cognition',C['cognition']),
            ('education_main','Education',C['grey']),('risk_logit','Total logit',C['prism'])]

def load(path): return pd.read_csv(path)

def subset(df, **filters):
    for k,v in filters.items(): df=df[df[k].eq(v)]
    return df

def one(df, **filters):
    rows=subset(df,**filters)
    assert len(rows)==1, (filters,len(rows))
    return rows.iloc[0]

def clean(ax, grid='x'):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    # Reference lines retain scientific meaning; decorative grids are omitted.
    ax.grid(False)

def legend(ax,handles,labels,ncol=2):
    return ax.legend(handles,labels,ncol=ncol,loc='lower center',bbox_to_anchor=(.5,1.065),
          borderaxespad=0,frameon=False,handlelength=1.6,columnspacing=.85,handletextpad=.4)

def handle(color,marker='o',ls='None',fill=True):
    return Line2D([],[],color=color,marker=marker,linestyle=ls,markersize=3.6,
                  markerfacecolor=color if fill else 'white',markeredgewidth=.7)

def interval(ax,row,y,color,marker='o',fill=True):
    ax.errorbar(row.beta,y,xerr=[[row.beta-row.ci_low],[row.ci_high-row.beta]],fmt=marker,
      color=color,ms=3.6,mfc=color if fill else 'white',mew=.65,lw=.8,capsize=1.8,zorder=3)

def stage_discrimination(ax):
    for j,(key,label,color,marker) in enumerate(METHODS):
        x=[one(metrics,task=t,horizon=h,model=key).auprc for t,h in ENDPOINTS]
        ax.scatter(x,np.arange(4)+(j-1.5)*.16,c=color,marker=marker,s=17,zorder=4,label=label,
                   edgecolors='white',linewidths=.3)
    for i,(task,h) in enumerate(ENDPOINTS):
        p=one(metrics,task=task,horizon=h,model='PRISM_current').weighted_prevalence
        ax.plot([p,p],[i-.37,i+.37],color=C['grey'],lw=.7,ls=':')
    ax.set(yticks=range(4),yticklabels=ENDLABELS,xlim=(0,1),ylim=(3.55,-.55),
           xlabel='Average precision (AP)',xticks=[0,.25,.5,.75,1])
    legend(ax,[handle(c,m) for _,_,c,m in METHODS],[n for _,n,_,_ in METHODS],ncol=4)
    clean(ax)

def memory_response(ax):
    for h,ls in [(3,'-'),(5,'--')]:
        for f in (1,2,3):
            d=subset(curves,model='PRISM_current',horizon=h,fold=f,feature='PHC_MEM_z').sort_values('raw_value')
            ax.plot(d.raw_value,d.centered_main_logit,color=C['prism'],ls=ls,alpha=.45+.15*f,lw=.85)
    ax.axhline(0,lw=.6,color=C['grey'],ls=':')
    ax.set(xlabel='Memory (PHC harmonized score)',ylabel='Centered main contribution (logit)',
           ylim=(-.85,.85),yticks=[-.8,-.4,0,.4,.8])
    legend(ax,[handle(C['prism'],marker='',ls='-'),handle(C['prism'],marker='',ls='--')],['3-year','5-year'])
    clean(ax,grid='y')

def feature_direction(ax):
    matrix=np.array([[one(stability,model='PRISM_current',horizon=h,feature=k)['delta_fold'+str(f)]
                for h in (3,5) for f in (1,2,3)] for k,_ in FEATURES])
    cmap=LinearSegmentedColormap.from_list('signed',['#4277A5','#FFFFFF','#CA802D'])
    norm=TwoSlopeNorm(0,vmin=-1.5,vmax=1.5)
    ax.pcolormesh(np.arange(7)-.5,np.arange(8)-.5,matrix,
                  cmap=cmap,norm=norm,rasterized=False)
    ax.set(xlim=(-.5,5.5),ylim=(6.5,-.5))
    for i in range(7):
        for j in range(6):
            rgb=np.array(cmap(norm(matrix[i,j]))[:3])
            linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
            luminance=float(linear @ np.array([.2126,.7152,.0722]))
            black_contrast=(luminance+.05)/.05
            white_contrast=1.05/(luminance+.05)
            assert max(black_contrast,white_contrast)>=4.5
            ax.text(j,i,format(matrix[i,j],'+.2f'),ha='center',va='center',fontsize=6.5,
                    color='black' if black_contrast>=white_contrast else 'white')
    ax.axvline(2.5,color='white',lw=3)
    ax.set(yticks=range(7),yticklabels=[v for _,v in FEATURES],xticks=range(6),
       xticklabels=['1','2','3','1','2','3'],xlabel='Development fold',ylabel='')
    ax.text(1,1.035,'3-year',transform=ax.get_xaxis_transform(),ha='center',fontsize=6.5)
    ax.text(4,1.035,'5-year',transform=ax.get_xaxis_transform(),ha='center',fontsize=6.5)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)

def paired_discrimination(ax):
    for j,(ref,name,color,marker) in enumerate(METHODS[1:]):
        for i,(task,h) in enumerate(ENDPOINTS):
            r=one(paired,model='PRISM_current',reference=ref,metric='auprc',task=task,horizon=h)
            ax.errorbar(r.difference,i+(j-1)*.21,xerr=[[r.difference-r.ci_low],[r.ci_high-r.difference]],
                 fmt=marker,color=color,ms=3.1,mfc=color,lw=.7,capsize=1.5)
    ax.axvline(0,color=C['grey'],lw=.7,ls=':')
    ax.set(yticks=range(4),yticklabels=ENDLABELS,ylim=(3.6,-.6),xlim=(-.14,.14),
           xticks=[-.1,0,.1],xlabel='AP difference (PRISM-AD − comparator)')
    legend(ax,[handle(c,m) for _,_,c,m in METHODS[1:]],[n for _,n,_,_ in METHODS[1:]],3)
    clean(ax)

def risk_distribution(ax):
    for key,name,color,ls in GROUPS:
        d=subset(cdf,group=key).sort_values('threshold')
        assert d.cdf.iloc[-1]==1 and (d.cdf.diff().dropna()>=0).all()
        ax.plot(d.threshold,d.cdf,color=color,ls=ls,lw=1,label=name+' (n='+str(int(d.n.iloc[0]))+')')
    ax.set(xlim=(0,1),ylim=(0,1.02),xticks=[0,.25,.5,.75,1],yticks=[0,.25,.5,.75,1],
           xlabel='Predicted 3-year MCI risk',ylabel='Cumulative proportion')
    legend(ax,[handle(c,marker='',ls=ls) for _,_,c,ls in GROUPS],
           [name for _,name,_,_ in GROUPS],3)
    clean(ax,grid='y')

def component_contrast(ax,right):
    for i,(component,label,color) in enumerate(COMPONENTS):
        for h,dy,m in [(3,-.12,'o'),(5,.12,'s')]:
            r=one(contrasts,model='PRISM_current',horizon=h,scope='primary_observed',component=component,right=right)
            assert r.status=='ok'
            interval(ax,r,i+dy,color,m)
    ax.axvline(0,color=C['grey'],lw=.7,ls=':')
    ax.set(yticks=range(4),yticklabels=[n for _,n,_ in COMPONENTS],ylim=(3.55,-.55),
       xlim=(-.85,.85),xticks=[-.8,-.4,0,.4,.8],xlabel='Adjusted difference (logit)')
    legend(ax,[handle(C['grey'],'o'),handle(C['grey'],'s')],['3-year','5-year'],2)
    clean(ax)

def later_ad_contrast(ax): component_contrast(ax,'CN_to_AD')

def other_cn_contrast(ax): component_contrast(ax,'other_CN_no_AD')


def audit_figure(fig,axes,letters):
    axes=list(fig.axes)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); bounds=fig.bbox
    # Structural text only: axis tick labels outside the view limits are not rendered.
    texts=[]
    for ax in axes:
        texts += [ax.xaxis.label,ax.yaxis.label]+list(ax.texts)
        for axis,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            for tick in axis.get_major_ticks():
                if min(lim)-1e-8<=tick.get_loc()<=max(lim)+1e-8:
                    texts += [tick.label1,tick.label2]
        if ax.get_legend(): texts += ax.get_legend().get_texts()
    texts += list(fig.texts)
    texts=[t for t in texts if t.get_visible() and t.get_text().strip()]
    boxes=[(t,t.get_window_extent(renderer)) for t in texts]
    clipped=[]; collisions=[]; legends=[]
    for t,b in boxes:
        if b.x0<bounds.x0-.2 or b.y0<bounds.y0-.2 or b.x1>bounds.x1+.2 or b.y1>bounds.y1+.2:
            clipped.append(t.get_text())
    for i,(t,a) in enumerate(boxes):
        for u,b in boxes[i+1:]:
            # >0.4 px in both dimensions catches glyph-box overlap, not adjacency.
            if min(a.x1,b.x1)-max(a.x0,b.x0)>.4 and min(a.y1,b.y1)-max(a.y0,b.y0)>.4:
                collisions.append([t.get_text(),u.get_text()])
    for ax in axes:
        leg=ax.get_legend()
        if leg:
            lb=leg.get_window_extent(renderer)
            for other in axes:
                if lb.overlaps(other.get_window_extent(renderer)): legends.append('legend touches data rectangle')
    return dict(size_mm=[round(v*25.4,3) for v in fig.get_size_inches()],
      text_size_pt_min=min(t.get_fontsize() for t in texts),
      text_size_pt_max=max(t.get_fontsize() for t in texts),clipped_text=clipped,
      text_bbox_collisions=collisions,legend_data_overlaps=legends,
      panel_label_x=[round(t.get_position()[0],4) for t in letters],
      caveat='Bounding boxes supplement, not replace, direct visual inspection of PDF renders.')

metrics = load(DATA / 'first_visit_metrics_recomputed.csv')

paired = load(DATA / 'current_model_paired_comparisons.csv')

curves = load(DATA / 'training_supported_shape_curves.csv')

stability = load(DATA / 'shape_stability.csv')

contrasts = load(DATA / 'adjusted_component_contrasts.csv')

cdf = load(DATA / 'clinical_origin_risk_cdf.csv')



