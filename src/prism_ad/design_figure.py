"""Revised quantitative contribution schematic; no numerical predictions."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.transforms import Bbox
from .figure_style import apply_figure_style
apply_figure_style(frame="none",font="DejaVu Sans",sizes=(9,8,7))
plt.rcParams.update({"savefig.bbox":None,"svg.fonttype":"none","svg.hashsalt":"prism-quantified-20261016",
    "figure.facecolor":"white","savefig.facecolor":"white"})
W,H=183,180
C=dict(ink="#202124",muted="#59636E",line="#CDD4DA",clinical="#D8892B",prs="#7656A4",
       prism="#087F78",clinical_fill="#FDF8F1",prs_fill="#F8F5FC",prism_fill="#F1F8F7")
TEXTS,CONTAINERS,ROUTES=[],[],[]
def label(x,y,s,sz=8,col="ink",ha="left",weight="normal",bounds=None):
    t=ax.text(x,y,s,fontsize=sz,color=C.get(col,col),ha=ha,va="center",linespacing=1.23,
        fontweight=weight,zorder=5)
    TEXTS.append(t)
    if bounds:CONTAINERS.append((t,bounds))
    return t
def box(x,y,w,h,s=None,col="line",fill="white",dashed=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=1.2",
        fc=C.get(fill,fill),ec=C.get(col,col),lw=.8,linestyle=(0,(3,2)) if dashed else "-",zorder=1))
    if s:label(x+w/2,y+h/2,s,ha="center",bounds=(x,y,w,h))
def arrow(points,col="muted"):
    for n,(a,b) in enumerate(zip(points,points[1:])):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>" if n==len(points)-2 else "-",
            mutation_scale=7,lw=.85,color=C.get(col,col),shrinkA=0,shrinkB=0,zorder=2))
        ROUTES.append((a,b))
def heading(letter,y,s):
    label(5,y,letter,10,weight="bold");label(11,y,s,9)
def draw():
    heading("a",174,"Explicit clinical and genetic contributions to predicted risk")
    box(5,153,122,12,col="clinical",fill="clinical_fill")
    label(66,161,"Clinical and study context",ha="center",bounds=(5,153,122,12))
    label(66,156,"Cognitive scores, age, education and other covariates",7,"muted",ha="center",bounds=(5,153,122,12))
    box(135,153,43,12,"Polygenic risk score\n(PRS)",col="prs",fill="prs_fill")
    box(5,131,58,14,"Clinical feature\nfunctions",col="clinical",fill="clinical_fill")
    box(69,131,58,14,"Selected clinical\npairwise terms",col="clinical",fill="clinical_fill")
    box(135,131,43,14,"Linear PRS term\nCoefficient ≥0",col="prs",fill="prs_fill")
    for x,col in [(34,"clinical"),(98,"clinical"),(156.5,"prs")]:arrow([(x,153),(x,145)],col)
    box(5,112,173,11,col="prism",fill="prism_fill")
    label(91.5,117.5,"Intercept + clinical functions + pairwise terms + PRS term",ha="center",bounds=(5,112,173,11))
    for x,col in [(34,"clinical"),(98,"clinical"),(156.5,"prs")]:arrow([(x,131),(x,123)],col)
    box(5,93,59,11,"Predicted log odds",col="prism")
    box(80,93,32,11,"Sigmoid",col="prism")
    box(129,93,49,11,"Predicted risk\n(probability)",col="prism",fill="prism_fill")
    arrow([(34,112),(34,104)],"prism");arrow([(64,98.5),(80,98.5)],"prism")
    arrow([(112,98.5),(129,98.5)],"prism")
    box(5,78,173,9,col="prism",dashed=True)
    label(91.5,82.5,"Optional earlier cognition adds a history contribution to the same sum",
        7,"muted",ha="center",bounds=(5,78,173,9))
    ax.plot([5,178],[73,73],lw=.65,color=C["line"])
    heading("b",68,"Matched-input comparisons with independently fitted prediction models")
    box(5,44,46,16)
    label(28,56,"Clinical inputs",ha="center",bounds=(5,44,46,16))
    label(28,49,"With or without PRS",7,"muted",ha="center",bounds=(5,44,46,16))
    box(64,44,62,16)
    label(95,56.5,"PRISM-AD · logistic regression",ha="center",bounds=(64,44,62,16))
    label(95,51.5,"HGB · TabPFN",ha="center",bounds=(64,44,62,16))
    label(95,47,"Independent model fits",7,"muted",ha="center",bounds=(64,44,62,16))
    arrow([(51,52),(64,52)])
    box(139,44,39,16)
    label(158.5,56,"CN → first MCI",ha="center",bounds=(139,44,39,16))
    label(158.5,51.5,"MCI → AD dementia",ha="center",bounds=(139,44,39,16))
    label(158.5,47,"Separate 3- and 5-year fits",7,"muted",ha="center",bounds=(139,44,39,16))
    arrow([(126,52),(139,52)])
    label(5,38.5,"Same task-specific participants and splits; three participant-separated development–validation folds",7,"muted")
    ax.plot([5,178],[33.5,33.5],lw=.65,color=C["line"])
    heading("c",28.5,"Clinical-course applications and older-adult phenotype context")
    box(5,4,84,19,col="prism")
    label(47,19.5,"Model application: observed clinical-course groups",ha="center",bounds=(5,4,84,19))
    label(47,13.5,"High-PRS CN · other CN · CN later AD",7,ha="center",bounds=(5,4,84,19))
    label(47,8,"Groups defined from PRS and diagnoses, without predictions",7,"muted",ha="center",bounds=(5,4,84,19))
    box(96,4,82,19,col="clinical")
    label(137,19.5,"Separate phenotype comparison at age ≥75 years",ha="center",bounds=(96,4,82,19))
    label(137,13.5,"High-PRS CN versus AD",7,ha="center",bounds=(96,4,82,19))
    label(137,8,"Measured education",7,"muted",ha="center",bounds=(96,4,82,19))
def geometric_qc():
    fig.canvas.draw();r=fig.canvas.get_renderer()
    items=[(t,t.get_window_extent(r)) for t in TEXTS]
    overlaps=[[a.get_text(),b.get_text()] for i,(a,ba) in enumerate(items) for b,bb in items[i+1:] if ba.overlaps(bb)]
    outside=[t.get_text() for t,b in items if not (fig.bbox.contains(b.x0,b.y0) and fig.bbox.contains(b.x1,b.y1))]
    clipping=[]
    for t,(x,y,w,h) in CONTAINERS:
        b=t.get_window_extent(r);bb=Bbox.from_extents(x,y,x+w,y+h).transformed(ax.transData)
        if not (bb.contains(b.x0,b.y0) and bb.contains(b.x1,b.y1)):clipping.append(t.get_text())
    hits=[]
    for a,b in ROUTES:
        p,q=ax.transData.transform([a,b])
        for t,bb in items:
            if ((abs(p[0]-q[0])<.01 and bb.x0<p[0]<bb.x1 and max(min(p[1],q[1]),bb.y0)<min(max(p[1],q[1]),bb.y1)) or
                (abs(p[1]-q[1])<.01 and bb.y0<p[1]<bb.y1 and max(min(p[0],q[0]),bb.x0)<min(max(p[0],q[0]),bb.x1))):hits.append(t.get_text())
    return dict(text_text_collisions=overlaps,out_of_canvas=outside,box_clipping=clipping,
        connector_text_hits=hits,min_font_pt=min(t.get_fontsize() for t in TEXTS),
        width_mm=W,height_mm=H,png_dpi=600,
        panel_bounds_mm={"a":[0,75,W,H],"b":[0,35,W,73],"c":[0,0,W,33.5]},
        review="Geometric self-QC; not independent review.")
def build():
    global fig,ax
    TEXTS.clear();CONTAINERS.clear();ROUTES.clear()
    apply_figure_style(frame="none",font="DejaVu Sans",sizes=(9,8,7))
    plt.rcParams.update({"savefig.bbox":None,"svg.fonttype":"none","svg.hashsalt":"prism-quantified-20261016",
        "figure.facecolor":"white","savefig.facecolor":"white"})
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=600)
    ax=fig.add_axes([0,0,1,1],xlim=(0,W),ylim=(0,H));ax.axis("off");draw();qc=geometric_qc()
    assert not any(qc[k] for k in ("text_text_collisions","out_of_canvas","box_clipping","connector_text_hits")),qc
    return fig,qc
