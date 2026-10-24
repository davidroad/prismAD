"""Figure 5: regenerate every empirical panel from current controlled results."""
import os,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ..plot_records import build as frame
C=dict(ink="#24313B",muted="#65717E",cognition="#D8892B",prs="#7656A4",prism="#007F79",other="#99A4AE",line="#D2DADD")
def stars(q,p=1):
 return "***"if q<.001 else"**"if q<.01 else"*"if q<.05 else"(*)"if p<.05 else""
def build():
 data=Path(os.environ["PRISM_AD_SOURCE_DATA"])/"plotting"
 from .figure5_layout import LAYOUT
 fig=frame(LAYOUT);axes=fig.axes
 examples=json.loads((data/"published_forecast_examples.json").read_text())["examples"]
 order=["resilience_CN","CN_to_AD","other_CN_no_AD"]
 for j,(ax,group)in enumerate(zip(axes[3:6],order)):
  ex=next(x for x in examples if x["group"]==group)
  v=[ex[k]for k in ["intercept","cognition_main","PRS","other_terms"]]
  cumulative=0
  for k,(value,col)in enumerate(zip(v,[C["muted"],C["cognition"],C["prs"],C["other"]])):
   prior=cumulative;cumulative+=value
   ax.add_patch(Rectangle((min(prior,cumulative),k-.30),abs(value),.60,facecolor=col,edgecolor="white",lw=.35,zorder=3))
   if k<3:ax.plot([cumulative,cumulative],[k+.30,k+1-.30],color=C["muted"],lw=.6,zorder=2)
   if k in [1,2]:ax.text(max(prior,cumulative)+.10,k,f"{value:+.2f}",fontsize=7,va="center",color=C["ink"])
  ax.barh(4,cumulative,height=.60,color=C["prism"],edgecolor="white",lw=.35,zorder=3)
  ax.axvline(0,lw=.6,color=C["muted"]);ax.axhline(3.5,lw=.5,color=C["line"])
  assert abs(cumulative-ex["total_log_odds"])<1e-6
  assert abs(1/(1+np.exp(-cumulative))-ex["predicted_probability"])<2e-7
  left=[11,71,131][j]/183
  fig.text(left,.956,f'3-year MCI risk: {100*ex["predicted_probability"]:.1f}%',fontsize=7,ha="left",va="center",color=C["prism"])
 profile=pd.read_csv(data/"cognitive_decile_profiles.csv")
 corr=pd.read_csv(data/"course_component_correlations.csv").set_index("group")
 for title,group in [("Overall","All"),("CN at last diagnosis","CN"),("MCI at last diagnosis","MCI"),("CN later AD","AD")]:
  candidates=[t for t in fig.texts if t.get_text()==title]
  if candidates:candidates[-1].set_text(title+f" (n = {int(corr.loc[group,'n'])})")
 tests=pd.read_csv(data/"Figure5_C_decile_AD_CN_tests.csv").fillna({"marker":""}).set_index("test")
 for i,(ax,group)in enumerate(zip(axes[7:],["All","CN","MCI","AD"])):
  z=profile[profile.group.eq(group)].sort_values("C_decile")
  for field,offset,color,label in [("cognition_factor",-.19,C["cognition"],"Cognition"),("PRS_factor",.19,C["prs"],"PRS")]:
   ax.bar(z.C_decile+offset,z[field],.36,color=color,edgecolor="none",label=label,zorder=3)
   sparse=z.n.lt(5)
   if sparse.any():ax.scatter(z.loc[sparse,"C_decile"]+offset,z.loc[sparse,field],s=12,marker="o",facecolor="white",edgecolor=color,lw=.7,zorder=4)
  ax.axhline(1,color=C["muted"],lw=.6,ls=(0,(2,2)),zorder=1)
  rho=corr.loc[group,"spearman_G_C"]
  label=f"ρ(G, C) = {rho:.2f}".replace("-","−")
  ax.text(.03,.95,label,transform=ax.transAxes,ha="left",va="top",fontsize=8,color=C["muted"])
  if group=="AD":
   ax.text(.03,.75,"AD vs CN: "+tests.loc["global","marker"],transform=ax.transAxes,ha="left",va="top",fontsize=7,color="#3B4249")
   for row in z.itertuples():
    marker=tests.loc["C_decile_"+str(row.C_decile),"marker"]
    if marker:ax.text(row.C_decile+.19,row.PRS_factor+.13,marker,ha="center",va="bottom",fontsize=7,color="#3B4249")
  # Shared limit adapts only if the new factors need more space.
  top=max(3.85,float(profile[["cognition_factor","PRS_factor"]].max().max())+.45)
  ax.set_ylim(0,top)
  ax.legend(loc="lower center",bbox_to_anchor=(.5,1.03),frameon=False,ncol=2,fontsize=8,handlelength=.9,handletextpad=.35,columnspacing=1,borderaxespad=0)
 # Cumulative forecast distributions in the independently defined targeted groups.
 ax=axes[0];cdf=pd.read_csv(data/"clinical_origin_risk_cdf.csv")
 for group,label,color in [("resilience_CN","High-PRS, last CN",C["prs"]),("CN_to_AD","CN later AD","#C75058"),("other_CN_no_AD","Other CN","#5792BB")]:
  q=cdf[cdf.group.eq(group)].sort_values("threshold")
  ax.plot(q.threshold,q.cdf,color=color,lw=1,label=label)
 ax.legend(loc="lower right",fontsize=6.5,frameon=False)
 # Schooling measurements retain all observations, with deterministic packed offsets.
 ax=axes[1];freq=pd.read_csv(data/"education_frequency.csv")
 for j,(group,color)in enumerate([("CN",C["prs"]),("AD","#C75058")]):
  q=freq[freq.status.eq(group)].sort_values("education")
  v=np.repeat(q.education.to_numpy(float),q.n.to_numpy(int));xs=[];ys=[]
  for row in q.itertuples(index=False):
   n=int(row.n);layers=int(np.ceil(n/65));counts=[n//layers+(k<n%layers)for k in range(layers)]
   for k,count in enumerate(counts):
    offsets=(np.arange(count)-(count-1)/2)*.010
    vertical=(k-(layers-1)/2)*.24
    xs.extend(j+offsets);ys.extend([row.education+vertical]*count)
  ax.scatter(xs,ys,s=.54,color=color,alpha=.92,linewidths=0,zorder=2)
  lo,mid,hi=np.quantile(v,[.25,.5,.75]);ax.plot([j+.40,j+.40],[lo,hi],color=color,lw=2,solid_capstyle="butt",zorder=3);ax.plot([j+.355,j+.445],[mid,mid],color=C["ink"],lw=1,zorder=4)
  ax.text(j,32,"n = "+str(len(v)),ha="center",fontsize=7)
 # Robust adjusted education effects, same two planned displays.
 ax=axes[2];educ=pd.read_csv(data/"education_tests.csv")
 for y,name,color in [(0,"primary_final75",C["prism"]),(1,"latest_record_confirmed",C["prs"])]:
  q=educ[educ.comparison.eq(name)].iloc[0]
  ax.errorbar(q.beta,y,xerr=[[q.beta-q.ci_low],[q.ci_high-q.beta]],fmt="o",ms=4,color=color,lw=.9,capsize=2)
  marker=stars(q.q,q.p)
  if marker:ax.text(q.ci_high+.07,y,marker,va="center",fontsize=8)
 ax.axvline(0,color=C["muted"],lw=.6,ls=":")
 rows=[educ[educ.comparison.eq(k)].iloc[0]for k in ["primary_final75","latest_record_confirmed"]]
 ax.set_yticks([0,1],["Last diagnosis\n(n = "+str(int(rows[0].n1+rows[0].n0))+")","Final visit CN/AD\n(n = "+str(int(rows[1].n1+rows[1].n0))+")"])
 # PET counts from the freshly redefined high-PRS sample.
 ax=axes[6];pet=json.loads((data/"pet_display.json").read_text())
 vals=[pet["positive"],pet["negative"]];total=sum(vals)
 for y,n,color in zip([1,0],vals,["#C75058","#99A4AE"]):
  ax.barh(y,n,height=.55,color=color)
  ax.text(n+1,y,f"{n}/{total} ({100*n/total:.1f}%)",va="center",fontsize=7)
 ax.set_yticks([0,1],["Amyloid-negative","Amyloid-positive"])
 ax.set_xlim(0,max(60,max(vals)+25))
 return fig
