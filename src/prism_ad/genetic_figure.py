"""Supplementary Figure 2: matched genetic-information increments."""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .figure_style import apply_figure_style
from .runtime import source_data
MODELS=[('PRISM_current','PRISM-AD'),('LR_current','Logistic'),('HGB_current','HGB'),('TabPFN_current','TabPFN')]
TASKS=[('CN_MCI','CN → MCI'),('MCI_AD','MCI → AD')]
def one(frame,**criteria):
    keep=np.ones(len(frame),bool)
    for key,value in criteria.items():keep &= frame[key].eq(value).to_numpy()
    result=frame.loc[keep]
    assert len(result)==1,criteria
    return result.iloc[0]
def build():
    apply_figure_style(font='Nimbus Sans',sizes=(7.5,7,6.5))
    plt.rcParams.update({'savefig.bbox':None,'svg.fonttype':'none',
        'svg.hashsalt':'prism-genetic-increment-20260918','axes.titlelocation':'center',
        'text.color':'#202124','axes.labelcolor':'#202124'})
    pairs=pd.read_csv(source_data()/'plotting/genetic_contrasts.csv')
    pairs['model']=pairs.model.map(lambda x:x+'_current')
    pairs['metric']=pairs.metric.map({'weighted_AUROC':'auroc','weighted_AP':'auprc','Hajek_Brier':'brier_hajek'})
    pairs=pairs[pairs['subset'].eq('first_eligible_landmark') & pairs.fold.astype(str).eq('pooled_123')]
    assert len(pairs[pairs.metric.isin(['auroc','auprc'])])==32
    colors={'PRISM_current':'#087F78','LR_current':'#303B46','HGB_current':'#68727D','TabPFN_current':'#C45A36'}
    outline={'panels':[]};records=[]
    fig=plt.figure(figsize=(183/25.4,132/25.4),dpi=180)
    rects=[[.15,.60,.31,.29],[.65,.60,.31,.29],
           [.15,.14,.31,.29],[.65,.14,.31,.29]]
    axes=[];letters=[]
    for row,(task,label) in enumerate(TASKS):
        for col,metric in enumerate(['auroc','auprc']):
            idx=2*row+col;letter='abcd'[idx];ax=fig.add_axes(rects[idx]);axes.append(ax)
            for y,(model,name) in enumerate(MODELS):
                for horizon,dy,marker in [(3,-.16,'o'),(5,.16,'s')]:
                    r=one(pairs,task=task,horizon=horizon,model=model,metric=metric)
                    assert r.ci_low<=r.ci_high and np.isfinite([r.difference,r.ci_low,r.ci_high]).all()
                    yy=y+dy;color=colors[model]
                    ax.plot([r.ci_low,r.ci_high],[yy,yy],color=color,lw=.85)
                    ax.plot(r.difference,yy,marker=marker,ms=3.4,color=color,
                            mfc=color if horizon==3 else 'white',mew=.85)
                    records.append({'panel':letter,'task':task,'horizon':horizon,'model':model,'metric':metric,
                        'difference':r.difference,'ci_low':r.ci_low,'ci_high':r.ci_high})
            limits=pairs[pairs.metric.eq(metric)]
            lo=min(0,float(limits.ci_low.min()));hi=max(0,float(limits.ci_high.max()))
            pad=max((hi-lo)*.09,.004)
            ax.set_xlim(lo-pad,hi+pad)
            ax.axvline(0,color='#8A929A',lw=.7,ls=':',zorder=0)
            ax.set_ylim(3.5,-.5)
            ax.set_yticks(range(4),[name for _,name in MODELS])
            ax.tick_params(axis='y',length=0,pad=5)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#59636E')
            ax.set_xlabel('ΔAUROC' if metric=='auroc' else 'ΔAP',labelpad=4)
            ticks=matplotlib.ticker.MaxNLocator(4).tick_values(lo-pad,hi+pad)
            ax.set_xticks([t for t in ticks if lo-pad<=t<=hi+pad])
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
            title=label+(' · discrimination' if col==0 else ' · precision–recall')
            title_y=.970 if row==0 else .495
            fig.text(rects[idx][0]+rects[idx][2]/2,title_y,title,ha='center',va='top',fontsize=7.5)
            letters.append(fig.text(.012 if col==0 else .512,title_y,letter,ha='left',va='top',fontsize=9,weight='bold'))
            outline['panels'].append({'letter':letter,'role':'primary' if row==0 else 'supporting',
                'message':title,'chart_family':'paired-difference interval plot','row':row,'col':col*6,
                'colspan':6,'data_vid':'genetic_contrasts.csv','label_budget':0,
                'ask':'Show all four models at both horizons without selecting favorable contrasts.'})
    handles=[Line2D([],[],color='#59636E',marker='o',mfc='#59636E',ms=3.4,lw=0),
             Line2D([],[],color='#59636E',marker='s',mfc='white',ms=3.4,lw=0)]
    fig.legend(handles,['3 years','5 years'],loc='upper center',bbox_to_anchor=(.54,.935),
               ncol=2,handletextpad=.4,columnspacing=1.5,borderaxespad=0)
    fig.text(.54,.025,'Difference: phenotype + PRS minus phenotype only; positive = higher discrimination',
             ha='center',va='bottom',fontsize=7)

    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for col in [0,1]:
        left=min(axes[i].get_tightbbox(renderer).x0 for i in [col,col+2])/fig.bbox.width-.012
        for i in [col,col+2]:letters[i].set_x(max(.008,left))
    return fig,records
