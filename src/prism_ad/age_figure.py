"""Supplementary Figure 2: ages at the first eligible prediction visit."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Text
from .figure_style import apply_figure_style
TASKS=[('CN_MCI','CN → MCI'),('MCI_AD','MCI → AD')]
PALETTE={'clinical_states': {'CN': '#5B8DB8', 'MCI': '#D8A332', 'AD': '#B44848', 'unresolved_or_interval_ambiguous': '#8A929A'}, 'ink': {'primary': '#202124', 'secondary': '#59636E', 'grid': '#D9DEE3', 'uncertainty': '#B8C0C8'}}

def plot(data):
    apply_figure_style(font='Nimbus Sans',sizes=(8,7,6))
    plt.rcParams.update({'savefig.bbox':None,'svg.fonttype':'none','svg.hashsalt':'prism-prediction-age','axes.titlelocation':'center'})
    registry=PALETTE
    colors = registry['clinical_states']
    fig = plt.figure(figsize=(183/25.4, 140/25.4))
    panels = []; axes = []
    for i, (task, title) in enumerate(TASKS):
        for j, horizon in enumerate((3,5)):
            letter = 'abcd'[i*2+j]
            ax = fig.add_axes([.20+.49*j, .60-.41*i, .285, .285]); axes.append(ax)
            q = data[(data.task == task) & (data.horizon == horizon) & data.group.ne('all')]
            stats = []; names = []
            for _, row in q.iterrows():
                stats.append(dict(med=row.age_median, q1=row.age_q1, q3=row.age_q3,
                                  whislo=row.age_min, whishi=row.age_max, fliers=[]))
                names.append('%s (n=%d)' % (row.group_label, row.n))
            positions = np.arange(len(q))[::-1]
            art = ax.bxp(stats, positions=positions, widths=.46, vert=False,
                         patch_artist=True, showfliers=False, manage_ticks=False,
                         medianprops={'color':'#202124', 'linewidth':1.1},
                         whiskerprops={'color':'#59636E','linewidth':.65},
                         capprops={'color':'#59636E','linewidth':.65})
            for patch, (_, row) in zip(art['boxes'], q.iterrows()):
                state = ('MCI' if task == 'CN_MCI' else 'AD') if row.group == 'target' else (
                    'CN' if row.group == 'event_free' else ('AD' if row.group == 'competing_ad'
                    else 'unresolved_or_interval_ambiguous'))
                # No recorded AD after MCI does not mean CN: use a neutral
                # no-event color in both tasks, rather than the CN-state hue.
                color = registry['ink']['uncertainty'] if row.group == 'event_free' else colors[state]
                patch.set_facecolor(color); patch.set_alpha(.72)
                if row.group == 'unknown': patch.set_hatch('//')
                patch.set_edgecolor('#59636E'); patch.set_linewidth(.6)
            ax.axvline(75, color='#59636E', linestyle=':', linewidth=.8, zorder=0)
            ax.set(xlim=(44,86), ylim=(-.6, len(q)-.4), xticks=[45,55,65,75,85],
                   yticks=positions, yticklabels=names, xlabel='Age at prediction visit (years)')
            ax.tick_params(axis='y', length=0, labelsize=7, pad=5)
            ax.spines['left'].set_visible(False)
            ax.set_title('%s | %d years' % (title,horizon), loc='center', pad=12)
            # Left edge of each complete panel, including its outcome labels.
            fig.text(.012+.49*j, .923-.41*i, letter, size=9, weight='bold', va='center')
            panels.append(dict(panel_label=letter, task=task, horizon=horizon,
                source_data='age_summary.csv', role='Age composition of the validation sample',
                reader_question='How old were the groups at the prediction visit?',
                order_rationale='CN before MCI; three-year before five-year horizon',
                caution='Descriptive unweighted distribution, not age-adjusted performance'))
    fig.text(.5, .065, 'Box: middle 50%; center line: median; whiskers: minimum–maximum.',
             ha='center', size=7)
    fig.text(.5, .030, 'Dotted line: 75 years. Groups are defined by recorded outcomes within each horizon.',
             ha='center', size=7)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    texts = [(t,t.get_window_extent(renderer)) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    overlaps = [(a.get_text(),b.get_text()) for k,(a,ba) in enumerate(texts)
                for b,bb in texts[k+1:] if ba.overlaps(bb)]
    clipped = [t.get_text() for t,b in texts if not (fig.bbox.contains(b.x0,b.y0)
               and fig.bbox.contains(b.x1,b.y1))]
    assert not overlaps, overlaps
    assert not clipped, clipped
    return fig,dict(text_overlaps=overlaps,clipped_text=clipped,panel_count=4)
