"""Current manuscript figures: frozen source metrics, no fitting or new tests."""
from pathlib import Path
import os,json,numpy as np,pandas as pd
import matplotlib.pyplot as plt
from . import common
from ..publication_numbering import SUPPLEMENTARY_FIGURE_IDS

def _capture(name):
    captured={}
    def keep(fig,number,receipt,removed,signature):
        captured.update(fig=fig,receipt=receipt,signature=signature)
    common.export=keep
    getattr(common,name)()
    return captured

def _letters(fig,number):
    from .panel_layout import LAYOUTS
    layouts=LAYOUTS
    layout=layouts[str(number)];w,h=fig.get_size_inches()*25.4
    for record in layout.get('label_adjustments',[]):
        t=next(t for t in fig.texts if t.get_text()==record['text'])
        t.set_position(record['new_position'])
    for record in layout['offsets']:
        t=next(t for t in fig.texts if t.get_text()==record['panel']and t.get_fontweight()=='bold')
        x,y=record['new_letter_anchor_mm'];t.set_position((x/w,y/h))

def _old_axes_signature(axes):
    import hashlib
    data=[dict(lines=[(np.asarray(x.get_xdata()).tolist(),np.asarray(x.get_ydata()).tolist())for x in a.lines],
               collections=[np.ma.asarray(x.get_offsets()).filled(np.nan).tolist()for x in a.collections])for a in axes]
    return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=True).encode()).hexdigest()

def _append_row(fig,old_height,new_height,number,data):
    old=list(fig.axes);signature=_old_axes_signature(old)
    common.reflow(fig,old_height,new_height,lambda y:y,set())
    W=183.
    def axis(x,y,w,h):
        a=fig.add_axes([x/W,1-(y+h)/new_height,w/W,h/new_height])
        a.spines["top"].set_visible(False);a.spines["right"].set_visible(False)
        return a
    def header(letter,title,x,y,center=None):
        fig.text(x/W,1-y/new_height,letter,fontsize=10,weight="bold",va="center")
        fig.text((x+6 if center is None else center)/W,1-y/new_height,title,
                 fontsize=8,va="center",ha="left"if center is None else"center")
    if number==2:
        from .calibration_panels import draw
        rows=pd.read_csv(data/"plotting/calibration_pooled_first_visit.csv")
        header("e","Calibration: CN → first MCI",5,164)
        header("f","Calibration: MCI → AD",96,164)
        left=axis(29,172,56,44);right=axis(122,172,55,44)
        diagnostics=pd.read_csv(data/"plotting/calibration_slope_tests.csv")
        a=draw(left,rows,"CN_MCI",legend=True,diagnostics=diagnostics);b=draw(right,rows,"MCI_AD",legend=False,diagnostics=diagnostics)
    else:
        from .marginal_panels import draw
        rows=pd.read_csv(data/"plotting/marginal_component_distributions.csv")
        header("e","Cognitive contributions",3.0368,155,56.5)
        header("f","Genetic contributions",91.0368,155,144.5)
        left=axis(25,166,63,52);right=axis(113,166,63,52)
        a=draw(left,rows,"C",legend=False);b=draw(right,rows,"G",legend=False)
        right.tick_params(labelleft=True)
    assert _old_axes_signature(old)==signature,"Original panels changed in data space"
    return dict(original_four_axes_signature=signature,original_data_marks_unchanged=True,
                panels_use_upstream_posthoc_diagnostics=True,inference_computed_in_renderer=False,new_panel_summaries=[a,b])

def build(figure):
    data=Path(os.environ['PRISM_AD_SOURCE_DATA'])
    if figure=='1':
        native=common.load('current_architecture_figure',Path(__file__).parent/'figure1_native.py')
        return native.fig,native.qc
    if figure=='2':
        cap=_capture('performance');fig=cap['fig']
        annotations=cap['receipt']['significance_annotations']
        tests=pd.read_csv(data/'plotting/Figure2_PRS_increment_tests.csv')
        for r in tests.itertuples():
            marker='***'if r.q<.001 else'**'if r.q<.01 else'*'if r.q<.05 else'(*)'if r.p<.05 else''
            if marker:
                i=[('CN_MCI',3),('CN_MCI',5),('MCI_AD',3),('MCI_AD',5)].index((r.task,r.horizon))
                axis=fig.axes[2 if r.metric=='weighted_AUROC'else 3]
                axis.text(axis.get_xlim()[1]+.004,i,marker,ha='left',va='center',fontsize=7,color='#24313B',clip_on=False)
        extension=_append_row(fig,163,240,2,data)
        return fig,dict(common.geometry(fig),**extension)
    if figure=='3':
        cap=_capture('overall');fig=cap['fig'];_letters(fig,3)
        tests=pd.read_csv(data/'plotting/Figure3_cognitive_quintile_omnibus_tests.csv')
        for i,group in enumerate(['All','CN','MCI','AD']):
            row=tests[tests.group.eq(group)].iloc[0]
            marker='***'if row.q<.001 else'**'if row.q<.01 else'*'if row.q<.05 else'(*)'if row.p<.05 else''
            if marker:fig.axes[i].text(.97,.94,marker,transform=fig.axes[i].transAxes,ha='right',va='top',fontsize=8)
        extension=_append_row(fig,155,235,3,data)
        return fig,dict(common.geometry(fig),**extension)
    if figure=='4':
        cap=_capture('contrasts');fig=cap['fig'];_letters(fig,4)
        # The native contrast renderer annotates current q/P values.
        return fig,common.geometry(fig)
    if figure=='5':
        from .figure5_native import build as updated_figure5
        fig=updated_figure5()
        return fig,common.geometry(fig)
    if figure==SUPPLEMENTARY_FIGURE_IDS['age']:
        from ..age_figure import plot
        return plot(pd.read_csv(data/'plotting/age_summary.csv'))
    if figure==SUPPLEMENTARY_FIGURE_IDS['prs']:
        from ..genetic_figure import build as genetic
        fig,_=genetic()
        for t in list(fig.texts):
            if t.get_text().startswith('Difference: phenotype'):t.remove()
        tests=pd.read_csv(data/'plotting/Figure2_PRS_increment_tests.csv')
        for r in tests.itertuples():
            marker='***'if r.q<.001 else'**'if r.q<.01 else'*'if r.q<.05 else'(*)'if r.p<.05 else''
            if marker:
                axis=0 if r.task=='CN_MCI'else 2
                metric_offset=0 if r.metric=='weighted_AUROC'else 1
                axis+=metric_offset
                y=(-.16 if r.horizon==3 else .16)
                fig.axes[axis].text(r.ci_high_original500+.0015,y,marker,fontsize=7,ha='left',va='center',color='#202124',clip_on=False)
        return fig,common.geometry(fig)
    if figure==SUPPLEMENTARY_FIGURE_IDS['optimization']:
        from .learning_curves import build as losses
        fig=losses()
        return fig,common.geometry(fig)
    raise ValueError(figure)
