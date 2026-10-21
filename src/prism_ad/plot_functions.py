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


apply_figure_style(font='Nimbus Sans', sizes=(8,7.5,7), grid=False)

plt.rcParams.update({'font.family':'Nimbus Sans', 'font.size':8, 'text.color':'#202124',
    'axes.labelsize':8, 'axes.titlesize':8, 'xtick.labelsize':7, 'ytick.labelsize':7,
    'legend.fontsize':7, 'axes.linewidth':.65, 'axes.labelpad':3,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'svg.hashsalt':'prism-overview-current', 'savefig.bbox':None,
    'figure.facecolor':'white', 'axes.facecolor':'white', 'savefig.facecolor':'white'})

C = dict(prism='#087F78', hgb='#68727D', tab='#C45A36', lr='#303B46',
    cognition='#D8892B', prs='#7656A4', cn='#5B8DB8', ad='#B44848',
    ink='#202124', grey='#59636E', line='#CDD4DA', pale='#F3F5F6')

MM = 25.4

QA = {}

FEATURES=[('visit_age','Age'),('PHC_Education','Education'),('global_z','Global cognition'),
          ('PHC_MEM_z','Memory'),('PHC_EXF_z','Executive'),('PHC_LAN_z','Language'),('PHC_VSP_z','Visuospatial')]

def canvas(w,h):
    fig=plt.figure(figsize=(w/MM,h/MM),dpi=180)
    fig._boxes=[]; fig._routes=[]
    return fig

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out',length=2.5,width=.6,pad=2)
    ax.grid(False)

def separated_curves(fig,left,bottom,feature,curves):
    """Separate horizons; three identifiable, unaveraged fold functions each."""
    d=curves[(curves.model=='PRISM_current') & (curves.feature==feature)]
    reference=float(d.reference_raw_value.iloc[0])
    assert np.allclose(d.reference_raw_value,reference)
    low,high=float(d.centered_main_logit.min()),float(d.centered_main_logit.max())
    pad=(high-low)*.13
    axs=[]
    for h,offset in [(3,.113),(5,0)]:
        ax=fig.add_axes([left,bottom+offset,.315,.079])
        for fold,ls,marker in [(1,'-','o'),(2,'--','s'),(3,':','^')]:
            z=d[(d.horizon==h)&(d.fold==fold)].sort_values('raw_value')
            ax.plot(z.raw_value,z.centered_main_logit,color=C['prism'],ls=ls,lw=.85,
                    marker=marker,markevery=[5,15,25],ms=2.6,mfc='white',mew=.65)
        ax.axhline(0,color=C['grey'],lw=.55,ls=':')
        ax.axvline(reference,color=C['grey'],lw=.55,ls=':')
        ax.set_ylim(low-pad,high+pad)
        ax.margins(x=.03)
        ax.yaxis.set_major_locator(MaxNLocator(3,prune='both'))
        ax.xaxis.set_major_locator(MaxNLocator(4,prune='both'))
        ax.text(1.035,.5,f'{h} y',transform=ax.transAxes,fontsize=7,ha='left',va='center')
        if h==3: ax.tick_params(axis='x',labelbottom=False)
        clean(ax);axs.append(ax)
    label={'PHC_MEM_z':f'Memory (PHC; ref. {reference:.3f})',
           'global_z':f'Global cognition (PHC; ref. {reference:.3f})',
           'PHC_Education':f'Education (years; ref. {reference:g})'}[feature]
    axs[-1].set_xlabel(label,fontsize=7.5)
    ylab=fig.text(left-.075,bottom+.096,'Main contribution (logit)',rotation=90,
                  fontsize=8,ha='center',va='center')
    handles=[Line2D([],[],color=C['prism'],ls=ls,marker=m,ms=2.6,mfc='white',lw=.85,label=f'Fold {f}')
             for f,ls,m in [(1,'-','o'),(2,'--','s'),(3,':','^')]]
    axs[0].legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.055),ncol=3,
                  frameon=False,handlelength=1.5,columnspacing=.7,handletextpad=.4,borderaxespad=0)
    return axs,ylab

def all_features(ax,stability):
    d=stability[stability.model=='PRISM_current']
    for i,(key,label) in enumerate(FEATURES):
        for h,marker,dy in [(3,'o',-.13),(5,'s',.13)]:
            r=d[(d.feature==key)&(d.horizon==h)].iloc[0]
            vals=[r['delta_fold'+str(f)] for f in (1,2,3)]
            # Individual fits only: no horizontal range that resembles a CI.
            ax.scatter(vals,i+dy+np.array([-.04,0,.04]),s=13,marker=marker,facecolor='white' if h==5 else C['prism'],
                       edgecolor=C['prism'],lw=.7,zorder=3)
    ax.axvline(0,color=C['grey'],lw=.6,ls=':')
    ax.set(yticks=range(7),yticklabels=[label for _,label in FEATURES],ylim=(6.6,-.6),
           xlabel='High − low main contribution (logit)')
    ax.margins(x=.12)
    ax.set_xticks([-1.5,-.75,0,.75,1.5])
    clean(ax)

def group_contrast(ax,right,contrasts):
    d=contrasts[(contrasts.model=='PRISM_current') & (contrasts.scope=='primary_observed') & (contrasts.right==right)]
    for i,(key,label,color) in enumerate([('cognition_main','Cognition',C['cognition']),
               ('PRS','PRS',C['prs']),('education_main','Education',C['grey']),('risk_logit','Total logit',C['ink'])]):
        for h,marker,dy in [(3,'o',-.13),(5,'s',.13)]:
            rows=d[(d.component==key)&(d.horizon==h)]
            assert len(rows)==1,(key,list(d.component.unique()))
            r=rows.iloc[0]
            ax.errorbar(r.beta,i+dy,xerr=[[r.beta-r.ci_low],[r.ci_high-r.beta]],fmt=marker,
                        ms=3.5,mfc='white' if h==5 else color,color=color,lw=.8,capsize=2)
    ax.axvline(0,color=C['grey'],ls=':',lw=.65)
    ax.set(yticks=range(4),yticklabels=['Cognition','PRS','Education','Total logit'],
           ylim=(3.55,-.55),xlim=(-.8,1.15),xticks=[-.5,0,.5,1],
           xlabel='Adjusted difference (logit; nominal 95% CI)')
    ax.xaxis.label.set_fontsize(7)
    clean(ax)

def interpretation():
    curves=pd.read_csv(DATA/'training_supported_shape_curves.csv')
    stability=pd.read_csv(DATA/'shape_stability.csv')
    contrasts=pd.read_csv(DATA/'adjusted_component_contrasts.csv')
    fig=canvas(183,206)
    fig._row_titles=[];fig._row_xlabels=[]
    panel_axes=[]; extra_labels=[]
    titles=['Memory','Global cognition','Education','All continuous feature functions',
            'GRPa-PRS CN versus other CN','GRPa-PRS CN versus later AD']
    for i,title in enumerate(titles):
        left=.145+(i%2)*.495; bottom=.73-(i//2)*.33
        if i<3:
            axes,ylab=separated_curves(fig,left,bottom,['PHC_MEM_z','global_z','PHC_Education'][i],curves)
            extra_labels.append((i%2,ylab))
            ax=axes[-1]
        else:
            ax=fig.add_axes([left,bottom,.34,.192]);axes=[ax]
            if i==3: all_features(ax,stability)
            else: group_contrast(ax,'other_CN_no_AD' if i==4 else 'CN_to_AD',contrasts)
            handles=[Line2D([],[],color=C['ink'],marker=m,ls='None',ms=3.5,
                       mfc='white' if h==5 else C['ink'],label=f'{h}-year') for h,m in [(3,'o'),(5,'s')]]
            ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,
                      borderaxespad=0,handlelength=1.5,columnspacing=1,fontsize=7)
        title_artist=fig.text(left+(.1575 if i<3 else .17),bottom+.235,title,fontsize=8,ha='center',va='bottom')
        fig._row_titles.append((i//2,title_artist))
        fig._row_xlabels.append((i//2,ax.xaxis.label))
        panel_axes.append((i,axes,bottom))
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    # Align letters to the true content edge shared by each column.
    for col in (0,1):
        starts=[a.get_tightbbox(renderer).x0 for i,axes,_ in panel_axes if i%2==col for a in axes]
        starts += [t.get_window_extent(renderer).x0 for c,t in extra_labels if c==col]
        x=min(starts)/fig.bbox.width
        for i,axes,bottom in panel_axes:
            if i%2==col: fig.text(x,bottom+.235,chr(97+i),weight='bold',fontsize=10,ha='left',va='bottom')
    return fig

def geometry_check(fig,name):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes=[(t,t.get_window_extent(renderer)) for t in texts]
    collisions=[]; overflow=[]; border=[]; route_hits=[]
    for i,(a,aa) in enumerate(boxes):
        if not fig.bbox.contains(aa.x0,aa.y0) or not fig.bbox.contains(aa.x1,aa.y1): overflow.append(a.get_text())
        for b,bb in boxes[i+1:]:
            if aa.overlaps(bb): collisions.append([a.get_text(),b.get_text()])
    for t,ax,(x,y,w,h) in fig._boxes:
        bb=t.get_window_extent(renderer)
        ll,ur=ax.transData.transform([(x,y),(x+w,y+h)])
        pad=1.5*fig.dpi/72
        if not(ll[0]+pad<=bb.x0 and ll[1]+pad<=bb.y0 and ur[0]-pad>=bb.x1 and ur[1]-pad>=bb.y1):
            border.append(t.get_text())
    for ax,a,b in fig._routes:
        pa,pb=ax.transData.transform([a,b])
        for t,bb in boxes:
            # Test the connector centreline, not its huge rectangular bounding box.
            pts=np.linspace(pa,pb,80)
            if any(bb.contains(*p) for p in pts): route_hits.append(t.get_text())
    gaps=[]
    if hasattr(fig,'_row_titles'):
        for row in (0,1):
            lower=min(t.get_window_extent(renderer).y0 for r,t in fig._row_xlabels if r==row)
            upper=max(t.get_window_extent(renderer).y1 for r,t in fig._row_titles if r==row+1)
            gap=(lower-upper)*25.4/fig.dpi
            gaps.append(round(gap,3))
            if gap<4: border.append(f'Inter-row text clearance {gap:.2f} mm < 4 mm')
    result=dict(text_text_collisions=collisions,out_of_canvas=overflow,
                box_padding_violations=border,connector_text_hits=sorted(set(route_hits)),
                minimum_font_pt=min(t.get_fontsize() for t in texts),n_text=len(texts),inter_row_clearance_mm=gaps)
    QA[name]=result
    return not(collisions or overflow or border or route_hits)
