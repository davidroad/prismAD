"""Article Figures 2–4 from aggregate estimates; no model fitting."""
from types import SimpleNamespace
from matplotlib.text import Text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import plot_prognosis as old
from . import plot_functions as ov
from .figure_style import apply_figure_style
from .runtime import source_data
DATA=source_data()/'plotting'
ed=SimpleNamespace(effects=pd.read_csv(DATA/'education_tests.csv'),
    desc=pd.read_csv(DATA/'education_descriptives.csv'),freq=pd.read_csv(DATA/'education_frequency.csv'))
base=SimpleNamespace(ed=ed,tests=pd.read_csv(DATA/'paired_AP_tests.csv'))
# Match the article's final style after loading the component modules.
apply_figure_style(font='Nimbus Sans',sizes=(7.5,7,6.5),grid=False)
plt.rcParams.update({'savefig.bbox':None,'svg.fonttype':'none','svg.hashsalt':'prism-v2-20261013',
    'axes.titlelocation':'center','pdf.fonttype':42})
annotations=[]
def stars(value):
    if not np.isfinite(value):return ''
    return '***' if value<.001 else '**' if value<.01 else '*' if value<.05 else ''

def nominal(ax):
    for j,(ref,name,color,marker) in enumerate(old.METHODS[1:]):
        for i,(task,h) in enumerate(old.ENDPOINTS):
            r=old.one(old.paired,model='PRISM_current',reference=ref,metric='auprc',task=task,horizon=h)
            ax.errorbar(r.difference,i+(j-1)*.27,xerr=[[r.difference-r.ci_low],[r.ci_high-r.difference]],fmt=marker,color=color,ms=3.1,lw=.7,capsize=1.5)
    ax.axvline(0,color=old.C['grey'],lw=.7,ls=':')
    ax.set(yticks=range(4),yticklabels=old.ENDLABELS,ylim=(3.6,-.6),xlabel='AP difference (PRISM-AD − comparator)',xlim=(-.14,.205),xticks=[-.1,0,.1])
    old.legend(ax,[old.handle(c,m) for _,_,c,m in old.METHODS[1:]],[n for _,n,_,_ in old.METHODS[1:]],3)
    old.clean(ax)
    ax.axvline(.143,color='#D9DEE3',lw=.5)
    ax.text(.173,-.45,'Nominal P',ha='center',va='center',fontsize=6.5)
    for j,(ref,name,color,marker) in enumerate(old.METHODS[1:]):
        for i,(task,h) in enumerate(old.ENDPOINTS):
            r=old.one(base.tests,model='PRISM_current',reference=ref,task=task,horizon=h,subset='first_eligible_landmark')
            s=stars(r.p)
            if s:ax.text(.173,i+(j-1)*.27,s,ha='center',va='center',fontsize=7,fontweight='bold',color=color)
            annotations.append(dict(figure=2,panel='b',task=task,horizon=h,reference=ref,p=float(r.p),q=float(r.q),label=s,criterion='nominal P',family='28 first-visit AP comparisons; q reported in Supplementary Table 2'))

def q_marks(ax,right,components,dy,fig,panel):
    lo,hi=ax.get_xlim();ax.set_xlim(lo,hi+.70)
    for h,x in [(3,hi+.19),(5,hi+.49)]:
        ax.text(x,-.47,str(h)+' y',ha='center',va='center',fontsize=6.5)
    for i,key in enumerate(components):
        for h,offset in [(3,-dy),(5,dy)]:
            r=old.one(old.contrasts,model='PRISM_current',scope='primary_observed',right=right,component=key,horizon=h)
            s=stars(r.q_all_128)
            if s:ax.text(hi+(.19 if h==3 else .49),i,s,ha='center',va='center',fontsize=7,fontweight='bold',color=old.C['ink'])
            annotations.append(dict(figure=fig,panel=panel,component=key,right=right,horizon=h,p=float(r.p),q=float(r.q_all_128),label=s,criterion='BH q',family='128 contribution contrasts'))

def education_effect(ax):
    for y,key in enumerate(['primary_final75','latest_record_confirmed']):
        r=old.one(base.ed.effects,comparison=key)
        old.interval(ax,r,y,old.C['prs'],marker='o' if y==0 else 's')
        ax.text(1.56,y,stars(r.q),ha='center',va='center',fontsize=8,fontweight='bold',color=old.C['prs'])
        ax.text(.98,y+.23,f'{r.beta:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]',
            transform=ax.get_yaxis_transform(),ha='right',va='top',fontsize=6.5)
        annotations.append(dict(figure=4,panel='d',comparison=key,p=float(r.p),q=float(r.q),label=stars(r.q),criterion='BH q',family='four education comparisons'))
    ax.axvline(0,color=old.C['grey'],lw=.7,ls=':')
    ax.set(ylim=(1.9,-.6),xlim=(-.12,1.7),xticks=[0,.5,1,1.5],yticks=[0,1],
        yticklabels=['Last classified\nassessment','Final recorded\nvisit confirmed'],
        xlabel='Adjusted education difference (CN − AD, years)')
    old.clean(ax)

def education_swarm(ax):
    # Exact education frequencies: one dot per observed education value.
    # Horizontal spreading only, preserving every y value and both full samples.
    source=base.ed.freq
    swarm_receipt=[]
    for i,(group,color) in enumerate([('CN',old.C['prs']),('AD',old.C['ad'])]):
        f=source[source.status.eq(group)].sort_values('education')
        values=np.repeat(f.education.to_numpy(),f.n.to_numpy(int))
        r=old.one(base.ed.desc,comparison='primary_final75',scope='all_eligible',group=group)
        assert len(values)==int(r.n_education)
        np.testing.assert_allclose(values.mean(),r['mean'])
        xs=[];ys=[]
        for row in f.itertuples(index=False):
            n=int(row.n)
            # Deterministic tie packing within narrow display-only value bands.
            # Exact recorded years remain the sole inputs to all summaries.
            layers=int(np.ceil(n/65))
            counts=[n//layers+(k<n%layers) for k in range(layers)]
            for k,count in enumerate(counts):
                offsets=(np.arange(count)-(count-1)/2)*.010
                vertical=(k-(layers-1)/2)*.24
                xs.extend(i+offsets);ys.extend([row.education+vertical]*count)
        ax.scatter(xs,ys,s=.54,color=color,alpha=.92,linewidths=0,rasterized=False,zorder=2)
        ax.plot([i+.40,i+.40],[r.q25,r.q75],color=color,lw=2,solid_capstyle='butt',zorder=3)
        ax.plot([i+.355,i+.445],[r['median'],r['median']],color=old.C['ink'],lw=1,zorder=4)
        ax.text(i,32,'n = '+str(len(values)),ha='center',fontsize=7)
        swarm_receipt.append(dict(group=group,n=len(values),mean=float(values.mean()),median=float(np.median(values)),vertical_jitter=True,display_only_vertical_offset_max_years=.24,statistics_use_unshifted_years=True,subsampling=False,one_point_per_observation=True,max_tied_count=int(f.n.max())))
    ax.set(xlim=(-.48,1.58),ylim=(-1,34),yticks=[0,8,16,24,32],xticks=[0,1],xticklabels=['Last-known CN','AD dementia'],ylabel='Education (years)')
    old.clean(ax)

def labels(fig,axes,rects,titles,ys):
    letters=[]
    for i,(ax,title) in enumerate(zip(axes,titles)):
        y=ys[i//2]
        letters.append(fig.text(.01 if i%2==0 else .51,y,'abcdef'[i],ha='left',va='top',weight='bold',fontsize=9))
        fig.text(rects[i][0]+rects[i][2]/2,y,title,ha='center',va='top',fontsize=7.5)
    fig.canvas.draw();r=fig.canvas.get_renderer()
    for col in [0,1]:
        left=min(axes[i].get_tightbbox(r).x0 for i in range(col,len(axes),2))/fig.bbox.width-.01
        for i in range(col,len(axes),2):letters[i].set_x(max(.008,left))
    return letters

def fig2():
    fig=plt.figure(figsize=(183/25.4,145/25.4),dpi=180)
    rects=[[.135,.620,.325,.280],[.635,.620,.335,.280],[.135,.170,.325,.280],[.635,.170,.335,.280]]
    axes=[fig.add_axes(r) for r in rects]
    for ax,fn in zip(axes,[old.stage_discrimination,nominal_comparisons,old.memory_response,old.feature_direction]):fn(ax)
    cb=fig.colorbar(axes[3].collections[0],cax=fig.add_axes([.700,.072,.20,.012]),orientation='horizontal',ticks=[-1.5,0,1.5])
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.outline.set_visible(False)
    cb.set_label('High − low contribution (logit)',fontsize=6.5,labelpad=2)
    ll=labels(fig,axes,rects,['Stage-specific discrimination','Paired model comparisons','Memory response functions','Directions across fitted functions'],[.982,.532])
    return fig,axes,ll

def fig3():
    fig=ov.interpretation()
    axes=fig.axes
    for ax,right,panel in [(axes[-2],'other_CN_no_AD','e'),(axes[-1],'CN_to_AD','f')]:
        q_marks(ax,right,['cognition_main','PRS','education_main','risk_logit'],.13,3,panel)
    for t in fig.texts:
        t.set_text(t.get_text().replace('GRPa-PRS CN','High-PRS CN'))
    letters=[t for t in fig.texts if t.get_text() in list('abcdef')]
    return fig,axes,letters

def fig4():
    fig=plt.figure(figsize=(183/25.4,155/25.4),dpi=180)
    rects=[[.12,.635,.34,.245],[.66,.635,.315,.245],[.12,.125,.34,.285],[.66,.125,.315,.285]]
    axes=[fig.add_axes(r) for r in rects]
    old.risk_distribution(axes[0]);axes[0].get_legend().get_texts()[0].set_text('High-PRS CN')
    old.later_ad_contrast(axes[1]);q_marks(axes[1],'CN_to_AD',[x[0] for x in old.COMPONENTS],.12,4,'b')
    ed.education_distribution(axes[2]);education_effect(axes[3])
    ll=labels(fig,axes,rects,['Risk at the CN starting visit','High-PRS CN versus subsequent AD','Education within the high-PRS interval','Education difference after adjustment'],[.953,.473])
    fig.text(.5,.991,'Model application: CN at prediction, subsequent clinical course',ha='center',va='top',fontsize=7.5)
    fig.text(.5,.514,'Separate phenotype comparison: high PRS, last classified age ≥75 years',ha='center',va='top',fontsize=7.5)
    return fig,axes,ll

def replacements(fig,mapping):
    found=set()
    for t in fig.findobj(Text):
        if t.get_text() in mapping:
            old=t.get_text();t.set_text(mapping[old]);found.add(old)
    assert found==set(mapping),(found,set(mapping))
nominal_comparisons=nominal
ed.education_distribution=education_swarm

def build(n):
    fn={2:fig2,3:fig3,4:fig4}[n]
    fig,axes,letters=fn()
    if n==2:
        fig.set_size_inches(183/25.4,150/25.4)
        for ax,rect in zip(axes,[[.135,.590,.325,.270],[.635,.590,.335,.270],[.135,.150,.325,.250],[.635,.150,.335,.250]]):ax.set_position(rect)
        replacements(fig,{'Stage-specific discrimination':'Discrimination by clinical stage','Paired model comparisons':'Differences between algorithms','Memory response functions':'PRISM-AD: memory contribution','Directions across fitted functions':'PRISM-AD: contribution changes'})
        for t in fig.texts:
            if abs(t.get_position()[1]-.982)<1e-5:t.set_y(.954)
            if abs(t.get_position()[1]-.532)<1e-5:t.set_y(.508)
        fig.text(.5,.995,'Current clinical measures + PRS',ha='center',va='top',fontsize=7.5)
    elif n==3:
        fig.set_size_inches(183/25.4,195/25.4)
        for ax in axes:
            x,y,w,h=ax.get_position().bounds;ax.set_position([x,y*.96,w,h*.96])
        for t in fig.texts:
            x,y=t.get_position();t.set_position((x,y*.96))
        fig.text(.5,.993,'PRISM-AD: CN → MCI with current clinical measures + PRS',ha='center',va='top',fontsize=7.5)
        replacements(fig,{'Memory':'Memory contribution','Global cognition':'Global cognition contribution','Education':'Education contribution','All continuous feature functions':'Contribution changes across folds','High-PRS CN versus other CN':'High-PRS CN − other CN','High-PRS CN versus later AD':'High-PRS CN − CN later with AD'})
        for ax in axes:
            if ax.get_yticklabels():ax.set_yticks(ax.get_yticks(),[t.get_text().replace(' contribution','') for t in ax.get_yticklabels()])
    else:
        replacements(fig,{'Model application: CN at prediction, subsequent clinical course':'PRISM-AD risk profiles: current clinical measures + PRS','Separate phenotype comparison: high PRS, last classified age ≥75 years':'Separate phenotype comparison: high PRS, last classified age ≥75 years','Risk at the CN starting visit':'Predicted MCI risk at the CN visit','High-PRS CN versus subsequent AD':'High-PRS CN − CN later with AD','Education within the high-PRS interval':'Measured educational attainment','Education difference after adjustment':'Adjusted education difference'})
    return fig,axes,letters
