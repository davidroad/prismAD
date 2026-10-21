"""Current manuscript figures: frozen source metrics, no fitting or new tests."""
from pathlib import Path
import os,json,numpy as np,pandas as pd
import matplotlib.pyplot as plt
from . import common

def _capture(name):
    captured={}
    def keep(fig,number,receipt,removed,signature):
        captured.update(fig=fig,receipt=receipt,signature=signature)
    common.export=keep
    getattr(common,name)()
    return captured

def _letters(fig,number):
    layouts=json.loads((Path(os.environ['PRISM_AD_SOURCE_DATA'])/'plotting/panel_layout.json').read_text())
    layout=layouts[str(number)];w,h=fig.get_size_inches()*25.4
    for record in layout.get('label_adjustments',[]):
        t=next(t for t in fig.texts if t.get_text()==record['text'])
        t.set_position(record['new_position'])
    for record in layout['offsets']:
        t=next(t for t in fig.texts if t.get_text()==record['panel']and t.get_fontweight()=='bold')
        x,y=record['new_letter_anchor_mm'];t.set_position((x/w,y/h))

def build(figure):
    data=Path(os.environ['PRISM_AD_SOURCE_DATA'])
    if figure=='1':
        native=common.load('current_architecture_figure',Path(__file__).parent/'figure1_native.py')
        return native.fig,native.qc
    if figure=='2':
        cap=_capture('performance');fig=cap['fig']
        annotations=cap['receipt']['significance_annotations']
        texts=[t for t in fig.axes[1].texts if t.get_text()in ['*','**','***']]
        for t,r in zip(texts,[r for r in annotations if r['label']]):
            t.set_text('***'if r['q']<.001 else'**'if r['q']<.01 else'*')
        tests=pd.read_csv(data/'plotting/Figure2_PRS_increment_tests.csv')
        for r in tests.itertuples():
            if r.p<.05 and r.q>=.05:
                assert r.metric=='weighted_AUROC'
                i=[('CN_MCI',3),('CN_MCI',5),('MCI_AD',3),('MCI_AD',5)].index((r.task,r.horizon))
                fig.axes[2].text(.044,i,'(*)',ha='center',va='center',fontsize=7,color='#24313B',clip_on=False)
        return fig,common.geometry(fig)
    if figure=='3':
        cap=_capture('overall');fig=cap['fig'];_letters(fig,3)
        return fig,common.geometry(fig)
    if figure=='4':
        cap=_capture('contrasts');fig=cap['fig'];_letters(fig,4)
        for r in cap['receipt']['significance_annotations']:
            if r['p']<.05 and r['q']>=.05:
                assert(r['panel'],r['component'],r['horizon'])==('a','risk_logit',3)
                fig.axes[0].text(r['ci_high']+.045,3.45-.16,'(*)',fontsize=7,ha='left',va='center',color='#24313B')
        return fig,common.geometry(fig)
    if figure=='5':
        from ..plot_records import build as from_records
        fig=from_records(data/'plotting/Figure5_plot_records.json')
        return fig,common.geometry(fig)
    if figure=='S1':
        from ..age_figure import plot
        return plot(pd.read_csv(data/'plotting/age_summary.csv'))
    if figure=='S2':
        from ..genetic_figure import build as genetic
        fig,_=genetic()
        for t in list(fig.texts):
            if t.get_text().startswith('Difference: phenotype'):t.remove()
        tests=pd.read_csv(data/'plotting/Figure2_PRS_increment_tests.csv')
        for task,h,axis,y in [('CN_MCI',5,0,.16),('MCI_AD',3,2,-.16)]:
            r=tests[(tests.task==task)&(tests.horizon==h)&(tests.metric=='weighted_AUROC')].iloc[0]
            assert r.p<.05 and r.q>=.05
            fig.axes[axis].text(r.ci_high_original500+.0015,y,'(*)',fontsize=7,ha='left',va='center',color='#202124',clip_on=False)
        return fig,common.geometry(fig)
    if figure=='S3':
        from .learning_curves import build as losses
        fig=losses()
        return fig,common.geometry(fig)
    raise ValueError(figure)
