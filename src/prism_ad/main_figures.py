"""Revised main figures from unchanged aggregate estimates; no model fitting."""
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
from . import plot_prognosis as old
from . import _legacy_panels as original
from .figure_style import apply_figure_style
def apply_style():
    apply_figure_style(font='Nimbus Sans',sizes=(8,7,6.5),grid=False)
    plt.rcParams.update({'savefig.bbox':None,'svg.fonttype':'none','svg.hashsalt':'prism-ad-quantified-20261016','pdf.fonttype':42,'axes.titlelocation':'center'})
apply_style()
C=old.C;annotations=[];qa={};swarm_receipt=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stars(v):return '***' if v<.001 else '**' if v<.01 else '*' if v<.05 else ''
def canvas(w,h):return plt.figure(figsize=(w/25.4,h/25.4),dpi=180)
def letter(fig,s,x,y):return fig.text(x,y,s,fontsize=10,weight='bold',ha='left',va='top')
def title(fig,s,x,y):return fig.text(x,y,s,fontsize=8,ha='center',va='top')
def key(ax,y=1.035):
    handles=[Line2D([],[],color=C['ink'],marker=m,ls='None',ms=3.5,mfc='white' if h==5 else C['ink'],label=f'{h}-year') for h,m in [(3,'o'),(5,'s')]]
    ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,y),ncol=2,frameon=False,borderaxespad=0,handlelength=1.5,columnspacing=1,fontsize=7)
def fig2():
    fig=canvas(183,95)
    axes=[fig.add_axes([.135,.19,.325,.54]),fig.add_axes([.635,.19,.335,.54])]
    old.stage_discrimination(axes[0])
    original.annotations.clear();original.nominal(axes[1])
    for item in original.annotations:
        item['family']=item['family'].replace('Supplementary Table 2','Supplementary Table 4')
    annotations.extend(original.annotations)
    title(fig,'Current clinical measures + PRS',.5,.985)
    letters=[letter(fig,'a',.025,.915),letter(fig,'b',.525,.915)]
    title(fig,'Discrimination by clinical stage',.2975,.915)
    title(fig,'Paired differences between algorithms',.8025,.915)
    return fig,axes,letters
def group_contrast(ax,right,panel):
    components=[('cognition_main','Cognition',C['cognition'],0),('PRS','PRS',C['prs'],1),('education_main','Education',C['grey'],2),('risk_logit','Total prediction',C['ink'],3.45)]
    for component,label,color,y in components:
        for h,m,dy in [(3,'o',-.13),(5,'s',.13)]:
            r=old.one(old.contrasts,model='PRISM_current',scope='primary_observed',right=right,component=component,horizon=h)
            assert r.status=='ok'
            ax.errorbar(r.beta,y+dy,xerr=[[r.beta-r.ci_low],[r.ci_high-r.beta]],fmt=m,ms=3.5,mfc='white' if h==5 else color,color=color,lw=.8,capsize=2)
            s=stars(r.q_all_128)
            if s:ax.text(1.34 if h==3 else 1.64,y,s,ha='center',va='center',fontsize=7,weight='bold',color=C['ink'])
            annotations.append(dict(figure=3,panel=panel,component=component,right=right,horizon=h,p=float(r.p),q=float(r.q_all_128),label=s,criterion='BH q',family='128 contribution contrasts',estimate=float(r.beta),ci_low=float(r.ci_low),ci_high=float(r.ci_high)))
    ax.axvline(0,color=C['grey'],ls=':',lw=.65);ax.axhline(2.76,color='#D9DEE3',lw=.55)
    for x,s in [(1.34,'3 y'),(1.64,'5 y')]:ax.text(x,-.47,s,ha='center',va='center',fontsize=7)
    ax.set(yticks=[p[3] for p in components],yticklabels=[p[1] for p in components],ylim=(4.02,-.55),xlim=(-.8,1.85),xticks=[-.5,0,.5,1],xlabel='Adjusted difference (log-odds; nominal 95% CI)')
    ax.xaxis.label.set_fontsize(7);old.clean(ax);key(ax)
def fig3():
    fig,axes,letters=original.build(3)
    # Keep six panels and all original functions/points; replace only contrast display.
    for ax,right,panel in [(axes[-2],'other_CN_no_AD','e'),(axes[-1],'CN_to_AD','f')]:
        ax.clear();group_contrast(ax,right,panel)
    replacements={
        'Global cognition contribution':'Mean cognitive score contribution',
        'Global cognition':'Mean cognitive score',
        'Total logit':'Total prediction',
        'Main contribution (logit)':'Centered contribution (log-odds)',
        'High − low main contribution (logit)':'High − low contribution (log-odds)',
        'Global cognition (PHC; ref. 0.904)':'Mean cognitive score (ref. 0.904)',
        'Contribution changes across folds':'High–low contrasts across folds'}
    for t in fig.findobj(Text):
        oldlabel=t.get_text()
        if oldlabel in replacements:t.set_text(replacements[oldlabel])
        if t.get_text() not in list('abcdef'):
            # Three role sizes only; numeric tick marks 6.5, secondary 7, base 8.
            v=t.get_fontsize();t.set_fontsize(6.5 if v<=6.5 else 7 if v<=7 else 8)
    axes[6].set_yticks(range(7), ['Age','Education','Mean cognitive\nscore','Memory','Executive','Language','Visuospatial'])
    # Cognitive/education colors thread into the term contrast panels.
    for ax in axes[:4]:
        for line in ax.lines:
            if line.get_color()==C['prism']:line.set_color(C['cognition'])
        if ax.get_legend():
            for h in ax.get_legend().legendHandles:h.set_color(C['cognition'])
    for ax in axes[4:6]:
        for line in ax.lines:
            if line.get_color()==C['prism']:line.set_color(C['grey'])
        if ax.get_legend():
            for h in ax.get_legend().legendHandles:h.set_color(C['grey'])
    # Feature labels determine the seven point-pair colors in panel d.
    for i,collection in enumerate(axes[6].collections):
        color=C['grey'] if i//2<2 else C['cognition']
        collection.set_edgecolor(color);collection.set_facecolor('white' if i%2 else color)
    return fig,axes,letters
def education_swarm(ax):
    original.education_swarm(ax)
    # The original deterministic packing displays one point per recorded value.
    for group,collection in zip(['CN','AD'],ax.collections):
        f=original.base.ed.freq[original.base.ed.freq.status.eq(group)].sort_values('education')
        values=np.repeat(f.education.to_numpy(),f.n.to_numpy(int))
        assert len(collection.get_offsets())==len(values)
        swarm_receipt.append(dict(group=group,n=len(values),mean=float(values.mean()),median=float(np.median(values)),subsampling=False,one_point_per_observation=True,statistics_use_unshifted_years=True,display_only_vertical_offset_max_years=.24,coordinates_equal_frozen=True,plotted_offsets_sha256=hashlib.sha256(collection.get_offsets().tobytes()).hexdigest()))
def education_effect(ax):
    original.annotations.clear();original.education_effect(ax)
    for r in original.annotations:r['panel']='c'
    annotations.extend(original.annotations)
    ax.set_xlabel('Adjusted education difference\n(CN − AD, years)')
def fig4():
    fig=canvas(183,170)
    axes=[fig.add_axes([.15,.635,.79,.225]),fig.add_axes([.12,.12,.34,.29]),fig.add_axes([.66,.12,.315,.29])]
    old.risk_distribution(axes[0]);axes[0].get_legend().remove()
    axes[0].legend(handles=[old.handle(c,marker='',ls=ls) for _,_,c,ls in old.GROUPS],labels=['High-PRS CN (n = 67)','Other CN (n = 526)','CN later with AD (n = 95)'],loc='lower center',bbox_to_anchor=(.5,1.025),ncol=3,borderaxespad=0,frameon=False,handlelength=1.8,columnspacing=1.5,handletextpad=.4,fontsize=7)
    education_swarm(axes[1]);education_effect(axes[2])
    title(fig,'Model application: CN at prediction, observed clinical course',.5,.989)
    title(fig,'Predicted 3-year MCI risk',.545,.946)
    title(fig,'Separate phenotype comparison: high PRS, last classified age ≥75 years',.5,.515)
    title(fig,'Measured educational attainment',.29,.472)
    title(fig,'Adjusted education difference',.8175,.472)
    letters=[letter(fig,'a',.025,.946),letter(fig,'b',.025,.472),letter(fig,'c',.525,.472)]
    return fig,axes,letters
def build(n):
    """Return revised Figure 2, 3 or 4 with its axes and panel letters."""
    apply_style();annotations.clear();swarm_receipt.clear()
    return {2:fig2,3:fig3,4:fig4}[int(n)]()
