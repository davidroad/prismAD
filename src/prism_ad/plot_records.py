"""Portable, JSON-only plotting records for the published Figure 5.
Records contain drawable coordinates, units/labels and styling, never executable
objects, participant identifiers, private covariates, images or model weights.
The original statistical analyses are separate from this figure-redraw contract.
"""
import json
from pathlib import Path
import numpy as np

def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(tuple,list)):return [plain(x) for x in value]
    return value

def _transform(value,axis=None):
    if axis is None:return 'figure'
    for name,tr in [('data',axis.transData),('axes',axis.transAxes),
                    ('xaxis',axis.get_xaxis_transform()),('yaxis',axis.get_yaxis_transform())]:
        if value==tr:return name
    return 'label'

def _text(t,axis=None):
    return dict(text=t.get_text(),xy=list(t.get_position()),transform=_transform(t.get_transform(),axis),
        fontsize=t.get_fontsize(),fontfamily=t.get_fontfamily(),fontstyle=t.get_fontstyle(),
        fontweight=t.get_fontweight(),color=plain(t.get_color()),ha=t.get_ha(),va=t.get_va(),
        rotation=t.get_rotation(),visible=t.get_visible(),clip_on=t.get_clip_on(),
        zorder=t.get_zorder(),linespacing=t._linespacing)

def _line(v,axis=None):
    dash=getattr(v,'_unscaled_dash_pattern',None)
    if dash is None:dash=(getattr(v,'_us_dashOffset',0),getattr(v,'_us_dashSeq',None))
    return dict(x=plain(np.asarray(v.get_xdata())),y=plain(np.asarray(v.get_ydata())),
        transform=_transform(v.get_transform(),axis) if axis else 'data',
        color=plain(v.get_color()),linewidth=v.get_linewidth(),
        linestyle=plain(dash) if v.is_dashed() else v.get_linestyle(),
        marker=v.get_marker(),markersize=v.get_markersize(),markerfacecolor=plain(v.get_markerfacecolor()),
        markeredgecolor=plain(v.get_markeredgecolor()),markeredgewidth=v.get_markeredgewidth(),
        alpha=v.get_alpha(),zorder=v.get_zorder(),drawstyle=v.get_drawstyle(),
        markevery=plain(v.get_markevery()),solid_capstyle=v.get_solid_capstyle(),
        solid_joinstyle=v.get_solid_joinstyle(),dash_capstyle=v.get_dash_capstyle(),
        dash_joinstyle=v.get_dash_joinstyle(),clip_on=v.get_clip_on())

def _axis(axis):
    from matplotlib.collections import PathCollection,LineCollection
    from matplotlib.patches import Rectangle
    collections=[]
    for c in axis.collections:
        extra=dict(alpha=c.get_alpha(),zorder=c.get_zorder(),clip_on=c.get_clip_on())
        if isinstance(c,PathCollection):
            assert len(c.get_paths())==1
            path=c.get_paths()[0]
            collections.append(dict(kind='scatter',offsets=plain(np.ma.asarray(c.get_offsets()).filled(np.nan)),
                vertices=plain(path.vertices),codes=plain(path.codes),sizes=plain(c.get_sizes()),
                facecolors=plain(c.get_facecolors()),edgecolors=plain(c.get_edgecolors()),
                linewidths=plain(c.get_linewidths()),**extra))
        elif isinstance(c,LineCollection):
            collections.append(dict(kind='segments',segments=plain(c.get_segments()),
                colors=plain(c.get_colors()),linewidths=plain(c.get_linewidths()),
                linestyles=plain(c._us_linestyles),**extra))
        else:raise TypeError(type(c))
    rectangles=[]
    for r in axis.patches:
        assert isinstance(r,Rectangle)
        rectangles.append(dict(xy=list(r.get_xy()),width=r.get_width(),height=r.get_height(),
            facecolor=plain(r.get_facecolor()),edgecolor=plain(r.get_edgecolor()),
            linewidth=r.get_linewidth(),alpha=r.get_alpha(),zorder=r.get_zorder(),
            hatch=r.get_hatch(),clip_on=r.get_clip_on(),linestyle=r.get_linestyle()))
    legend=None;leg=axis.get_legend()
    if leg:
        handles=[]
        for h in leg.legendHandles:
            if hasattr(h,'get_xdata'):handles.append(dict(kind='line',spec=_line(h)))
            elif isinstance(h,Rectangle):handles.append(dict(kind='rectangle',
                facecolor=plain(h.get_facecolor()),edgecolor=plain(h.get_edgecolor()),
                linewidth=h.get_linewidth()))
            else:raise TypeError(type(h))
        legend=dict(handles=handles,labels=[t.get_text()for t in leg.get_texts()],
            fontsize=leg._fontsize,text_styles=[_text(t,axis)for t in leg.get_texts()],loc=leg._loc,
            ncol=getattr(leg,'_ncol',getattr(leg,'_ncols',1)),frameon=leg.get_frame_on(),
            bbox_to_anchor=plain(leg._bbox_to_anchor._bbox.bounds)if leg._bbox_to_anchor is not None else None,
            **{k:getattr(leg,k)for k in ['handlelength','handletextpad','borderaxespad',
                'borderpad','labelspacing','columnspacing','markerscale']})
    decorations={}
    for key,a in [('x',axis.xaxis),('y',axis.yaxis)]:
        label=_text(a.label,axis)if a.label.get_text()else None
        if label:
            pixel=a.label.get_transform().transform(a.label.get_position())
            label['figure_xy']=plain(axis.figure.transFigure.inverted().transform(pixel))
        decorations[key]=dict(label=label,
            effective_ticks=[dict(pad=t.get_pad(),size=t.tick1line.get_markersize(),
                width=t.tick1line.get_markeredgewidth(),color=plain(t.tick1line.get_color()),
                markeredgecolor=plain(t.tick1line.get_markeredgecolor()),markerfacecolor=plain(t.tick1line.get_markerfacecolor()),
                label1=_text(t.label1,axis),label2=_text(t.label2,axis))for t in a.get_major_ticks()],
            labelpad=a.labelpad,ticks=plain(a.get_majorticklocs()),
            ticklabels=[t.label1.get_text()for t in a.get_major_ticks()],
            tick_params={k:plain(v)for k,v in a._major_tick_kw.items()
                if k in ['size','width','color','tickdir','pad','labelsize','labelcolor',
                          'tick1On','tick2On','label1On','label2On']})
    return dict(position=plain(axis.get_position().bounds),xlim=plain(axis.get_xlim()),
        ylim=plain(axis.get_ylim()),facecolor=plain(axis.get_facecolor()),axisbelow=axis.get_axisbelow(),
        axes=decorations,spines={k:dict(visible=s.get_visible(),color=plain(s.get_edgecolor()),
            linewidth=s.get_linewidth(),linestyle=s.get_linestyle())for k,s in axis.spines.items()},
        lines=[_line(v,axis)for v in axis.lines],collections=collections,rectangles=rectangles,
        texts=[_text(t,axis)for t in axis.texts],legend=legend)

def export_figure(fig,path,panels):
    fig.canvas.draw()
    record=dict(schema_version=1,figure_inches=plain(fig.get_size_inches()),figure_dpi=fig.dpi,
        facecolor=plain(fig.get_facecolor()),axes=[_axis(a)for a in fig.axes],
        texts=[_text(t)for t in fig.texts],axes_panels=panels,
        scope='Published display coordinates only; upstream model fitting and inference are separate.')
    assert len(record['axes'])==len(panels)
    Path(path).write_text(json.dumps(record,indent=2,allow_nan=False,default=plain)+'\n')
    return record

def _style_text(t,spec):
    for key in ['fontsize','fontfamily','fontstyle','fontweight','color','ha','va',
                'rotation','visible','clip_on','zorder','linespacing']:
        getattr(t,'set_'+key)(spec[key])

def _kwargs_line(spec):
    result={k:v for k,v in spec.items()if k not in ['x','y','transform']}
    if isinstance(result['linestyle'],list):
        result['linestyle']=(result['linestyle'][0],tuple(result['linestyle'][1]))
    return result

def build(path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import Rectangle,Patch
    from .figures_current.style import apply_figure_style
    apply_figure_style(font='Nimbus Sans',sizes=(9,8,7),grid=False)
    plt.rcParams.update({'savefig.bbox':None,'svg.fonttype':'none','pdf.fonttype':42})
    data=json.loads(Path(path).read_text())
    assert data['schema_version']==1 and len(data['axes'])==11
    fig=plt.figure(figsize=data['figure_inches'],dpi=data['figure_dpi'],facecolor=data['facecolor'])
    for spec in data['axes']:
        ax=fig.add_axes(spec['position'],facecolor=spec['facecolor'])
        ax.set_axisbelow(spec['axisbelow'])
        tr={'data':ax.transData,'axes':ax.transAxes,'xaxis':ax.get_xaxis_transform(),
            'yaxis':ax.get_yaxis_transform()}
        for rect in spec['rectangles']:ax.add_patch(Rectangle(**rect))
        for line in spec['lines']:
            ax.plot(line['x'],line['y'],transform=tr[line['transform']],**_kwargs_line(line))
        for c in spec['collections']:
            extra={k:c[k]for k in ['alpha','zorder','clip_on']}
            if c['kind']=='scatter':
                offsets=np.array(c['offsets'])
                ax.scatter(offsets[:,0],offsets[:,1],s=c['sizes'],
                    marker=MplPath(c['vertices'],c['codes']),facecolors=c['facecolors'],
                    edgecolors=c['edgecolors'],linewidths=c['linewidths'],**extra)
            else:
                linestyle=[(a,tuple(b)if b is not None else None)for a,b in c['linestyles']]
                ax.add_collection(LineCollection(c['segments'],colors=c['colors'],
                    linewidths=c['linewidths'],linestyles=linestyle,**extra))
        for label in spec['texts']:
            assert label['transform']!='label'
            t=ax.text(*label['xy'],label['text'],transform=tr[label['transform']])
            _style_text(t,label)
        for dim in ['x','y']:
            s=spec['axes'][dim];axis=getattr(ax,dim+'axis')
            getattr(ax,'set_'+dim+'ticks')(s['ticks'])
            getattr(ax,'set_'+dim+'ticklabels')(s['ticklabels'])
            mapping={'size':'length','tickdir':'direction','tick1On':'bottom'if dim=='x'else'left',
                'tick2On':'top'if dim=='x'else'right','label1On':'labelbottom'if dim=='x'else'labelleft',
                'label2On':'labeltop'if dim=='x'else'labelright'}
            axis.set_tick_params(**{mapping.get(k,k):v for k,v in s['tick_params'].items()})
            if s['label']:
                label=s['label']
                getattr(ax,'set_'+dim+'label')(label['text'],labelpad=s['labelpad'])
                _style_text(axis.label,label)
                axis.set_label_coords(*label['figure_xy'],transform=fig.transFigure)
            for tick,effective in zip(axis.get_major_ticks(),s['effective_ticks']):
                tick.set_pad(effective['pad'])
                for mark in [tick.tick1line,tick.tick2line]:
                    mark.set_markersize(effective['size']);mark.set_markeredgewidth(effective['width'])
                    mark.set_color(effective['color']);mark.set_markeredgecolor(effective['markeredgecolor']);mark.set_markerfacecolor(effective['markerfacecolor'])
                _style_text(tick.label1,effective['label1'])
                _style_text(tick.label2,effective['label2'])
        ax.set_xlim(spec['xlim']);ax.set_ylim(spec['ylim'])
        for name,s in spec['spines'].items():
            spine=ax.spines[name];spine.set_visible(s['visible']);spine.set_color(s['color'])
            spine.set_linewidth(s['linewidth']);spine.set_linestyle(s['linestyle'])
        if spec['legend']:
            s=spec['legend'].copy();text_styles=s.pop('text_styles');handles=[]
            for h in s.pop('handles'):
                if h['kind']=='line':handles.append(Line2D([],[],**_kwargs_line(h['spec'])))
                else:handles.append(Patch(**{k:v for k,v in h.items()if k!='kind'}))
            if s['bbox_to_anchor']is None:s.pop('bbox_to_anchor')
            legend=ax.legend(handles=handles,**s)
            for t,style in zip(legend.get_texts(),text_styles):_style_text(t,style)
    for spec in data['texts']:
        t=fig.text(*spec['xy'],spec['text']);_style_text(t,spec)
    return fig
