#!/usr/bin/env python3
"""Integrated, code-bound PRISM-AD workflow with typed inputs and explicit sums.
No model is fitted and no fitted response curve, neuron count, or result is drawn.
"""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle,FancyArrowPatch
from PIL import Image
HERE=Path(__file__).resolve().parent
STYLE=HERE/'style.py'
sp=importlib.util.spec_from_file_location('integrated_author_style',STYLE)
sty=importlib.util.module_from_spec(sp);sp.loader.exec_module(sty)
sty.apply_figure_style(font='Nimbus Sans',sizes=(8,7,7),grid=False)
plt.rcParams.update({'savefig.bbox':None,'svg.fonttype':'none','svg.hashsalt':'prism-integrated-figure1-20261019','figure.facecolor':'white','savefig.facecolor':'white'})
W,H=183,250
C={'ink':'#24313B','muted':'#65717E','line':'#C5CCD1','cognition':'#D8892B','prs':'#7656A4'}
fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=500)
ax=fig.add_axes([0,0,1,1],xlim=(0,W),ylim=(H,0));ax.axis('off')
TEXT=[];BOXES=[];ROUTES=[]
def color(c):return C.get(c,c)
def text(x,y,s,size=7,col='ink',ha='left',weight='normal'):
 t=ax.text(x,y,s,fontsize=size,color=color(col),ha=ha,va='center',weight=weight,linespacing=1.18,zorder=6);TEXT.append(t);return t
def line(points,col='muted',lw=.65,z=1):
 p=np.asarray(points);ax.plot(p[:,0],p[:,1],color=color(col),lw=lw,zorder=z)
def arrow(points,col='muted',lw=.75):
 if len(points)>2:line(points[:-1],col,lw)
 ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=6,lw=lw,color=color(col),shrinkA=0,shrinkB=0,zorder=2));ROUTES.append(points)
def module(x,y,w,h,s,size=7,col='ink',edge='line'):
 ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=color(edge),lw=.6,zorder=3))
 t=text(x+w/2,y+h/2,s,size,col,ha='center');BOXES.append((t,x,y,w,h));return t
def node(x,y,s,col='muted',r=3.1):
 ax.add_patch(Circle((x,y),r,facecolor='white',edgecolor=color(col),lw=.7,zorder=3))
 text(x,y,s,8,col,ha='center')
def title(letter,y,s):
 text(5,y,letter,10,weight='bold');text(11,y,s,8)
def brace(x,y0,y1,col='muted'):
 line([(x-1,y0),(x,y0),(x,y1),(x-1,y1)],col,.65)
# a. Clinical targets are defined before the model.
title('a',6,'Clinical transitions and prediction horizons')
module(7,13,14,8,'CN')
arrow([(21,17),(28,17)])
module(28,13,37,8,'First recorded MCI')
module(85,13,15,8,'MCI')
arrow([(100,17),(107,17)])
module(107,13,31,8,'AD dementia')
module(152,13,27,8,'3 and 5 years')
# b. Names and data types are separate from learned per-input functions.
title('b',27,'Clinical and genetic contributions to a complete prediction')
text(7,34,'Input',7,'muted')
text(53,34,'Input type',7,'muted',ha='center')
text(91,34,r'Per-input $f_j$',7,'muted',ha='center')
text(125,34,'Sum',7,'muted',ha='center')
cognitive=[('Memory',40),('Executive function',48),('Language',56),('Visuospatial',64),('Available-domain\nmean',72)]
for j,(label,y) in enumerate(cognitive,1):
 text(7,y,label,8)
 arrow([(67,y),(80,y)],'cognition')
 module(81,y-2.7,20,5.4,r'$f_{'+str(j)+r'}$',8,'cognition')
 arrow([(101,y),(113,y)],'cognition')
brace(42,37,75)
text(53,56,'Continuous',7,'muted',ha='center')
line([(114,40),(114,72)],'cognition',.7)
arrow([(114,56),(121.5,56)],'cognition')
node(125,56,r'$\Sigma$','cognition')
text(134,56,'C',8,'cognition',ha='center')
# Every other transformed clinical predictor still has a separate learned term.
clinical=[('Age',84,'Continuous',r'$f_6$'),('Education',92,'Continuous',r'$f_7$'),
          ('Sex / study',100,'Categorical',r'$f_j$'),('Missingness',108,'Binary',r'$f_j$'),
          ('Domain coverage',116,'Count',r'$f_j$')]
for label,y,dtype,fname in clinical:
 text(7,y,label,8)
 if y not in [84,92]:text(53,y,dtype,7,'muted',ha='center')
 arrow([(67,y),(80,y)])
 module(81,y-2.7,20,5.4,fname,8,'muted')
 arrow([(101,y),(113,y)])
brace(42,81,95)
text(53,88,'Continuous',7,'muted',ha='center')
text(7,124,'Clinical pairs',8)
text(53,124,'Continuous',7,'muted',ha='center')
arrow([(67,124),(80,124)])
module(81,121.3,20,5.4,r'$g_{jk}$',8,'muted')
arrow([(101,124),(113,124)])
line([(114,84),(114,124)],'muted',.7)
arrow([(114,104),(121.5,104)])
node(125,104,r'$\Sigma$','muted')
text(134,104,'R',8,'muted',ha='center')
text(125,121,r'$b+T$',7,'muted',ha='center')
arrow([(125,117.5),(125,107.5)])
# PRS is a separate linear path and has no edge into the clinical functions.
text(7,132,'AD PRS',8,'prs')
text(53,132,'Continuous',7,'muted',ha='center')
arrow([(67,132),(80,132)],'prs')
module(81,129.3,20,5.4,r'$\beta z$',8,'prs')
arrow([(101,132),(121,132)],'prs')
text(125,132,'G',8,'prs',ha='center')
# All three complete components enter the fitted log odds, then a sigmoid.
arrow([(137,56),(139,56),(139,89),(148.5,89)],'cognition')
arrow([(137,104),(138,104),(138,92),(147.5,92)])
arrow([(129,132),(141,132),(141,95),(148.5,95)],'prs')
node(151,92,r'$\Sigma$','ink',3.4)
text(151,103,r'$C+G+R$',7,ha='center')
arrow([(154.7,92),(161.5,92)],'ink')
module(162,87,10,10,r'$\sigma$',8,'ink')
text(167,81,'Sigmoid',7,ha='center')
arrow([(172,92),(176,92)],'ink')
text(179,92,r'$p$',8,ha='center')
# Compact key defines fj without enumerated units, hidden layers or fitted curves.
text(7,142,r'$f_j=$',8)
module(25,138.5,29,7,'Nonlinear neural term')
text(59.5,142,'+',8,ha='center')
module(65,138.5,27,7,'Linear term')
text(98,142,'+',8,ha='center')
module(104,138.5,44,7,'Eligible threshold term')
# c. History also supplies clinical summaries, not only the Transformer term.
title('c',150,'Optional cognitive-history information')
module(7,163,28,14,'Prior cognitive\nassessments')
arrow([(21,163),(21,160),(43,160)])
module(44,155,33,10,'History summaries')
arrow([(77,160),(83,160)])
module(84,155,35,10,'Clinical main /\npair terms')
arrow([(119,160),(126,160)])
text(131,160,r'$R$',8,ha='center')
arrow([(35,176),(43,176)])
module(44,171,33,10,'Transformer\nencoder')
arrow([(77,176),(83,176)])
module(84,171,31,10,'Domain pooling')
arrow([(115,176),(122,176)])
module(123,171,32,10,'Current-severity\ngate')
arrow([(155,176),(163,176)])
text(173,176,'T added to R',8,ha='center')
# d. Fitted preprocessing and training resampling never use held-out participants.
title('d',185,'Participant-separated development and evaluation')
module(7,197,27,14,'Participant split\n3 rounds')
arrow([(34,204),(40,204)])
module(41,197,36,14,'Training-only\npreprocessing')
arrow([(77,204),(83,204)])
module(84,197,33,14,'One shared\nweighted resample')
arrow([(117,204),(123,204)])
module(124,197,26,14,'Joint\noptimization')
arrow([(150,204),(156,204)])
module(157,197,21,14,'Held-out\nweighted\nevaluation')
arrow([(20.5,211),(20.5,221),(56,221)])
module(57,216,50,10,'Apply fitted preprocessing\nto held-out participants')
arrow([(59,211),(59,213),(82,213),(82,215.5)],'line',.65)
arrow([(107,221),(167.5,221),(167.5,211)])
# e. Term inspection and matched refits answer distinct, complementary questions.
title('e',230,'Two complementary uses of the fitted model')
module(7,239,27,8,'Inspect C, G, R')
arrow([(34,243),(40,243)])
module(41,239,45,8,'Clinical-course profiles')
module(99,239,29,8,'Refit ± PRS')
arrow([(128,243),(135,243)])
module(136,239,42,8,'Added prognostic\ninformation')
# Geometric verification
