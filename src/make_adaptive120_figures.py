"""Publication-style training, band-weight, CG and sector-summary figures."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'result'/'data'; FIG=ROOT/'result'/'figures'
RELEASE=ROOT/'result'/'data'/'local_v4_final'
RUNS={
 ('1/3','full'):RELEASE/'traces'/'nu1of3_full_m.json',
 ('1/3','gamma'):RELEASE/'traces'/'nu1of3_v4_gamma.json',
 ('1/3','outer0'):RELEASE/'traces'/'nu1of3_v4_outer_c3_p0.json',
 ('1/3','outer1'):RELEASE/'traces'/'nu1of3_v4_outer_c3_p1.json',
 ('1/3','outer2'):RELEASE/'traces'/'nu1of3_v4_outer_c3_p2.json',
 ('2/3','full'):RELEASE/'traces'/'nu2of3_full_m.json',
 ('2/3','gamma'):RELEASE/'traces'/'nu2of3_v4_gamma.json',
 ('2/3','outer0'):RELEASE/'traces'/'nu2of3_v4_outer_c3_p0.json',
 ('2/3','outer1'):RELEASE/'traces'/'nu2of3_v4_outer_c3_p1.json',
 ('2/3','outer2'):RELEASE/'traces'/'nu2of3_v4_outer_c3_p2.json'}
DIAG={
 ('1/3','full'):RELEASE/'diagnostics'/'nu1of3_full_m.npz',
 ('1/3','gamma'):RELEASE/'diagnostics'/'nu1of3_v4_gamma.npz',
 ('1/3','outer0'):RELEASE/'diagnostics'/'nu1of3_v4_outer_c3_p0.npz',
 ('1/3','outer1'):RELEASE/'diagnostics'/'nu1of3_v4_outer_c3_p1.npz',
 ('1/3','outer2'):RELEASE/'diagnostics'/'nu1of3_v4_outer_c3_p2.npz',
 ('2/3','full'):RELEASE/'diagnostics'/'nu2of3_full_m.npz',
 ('2/3','gamma'):RELEASE/'diagnostics'/'nu2of3_v4_gamma.npz',
 ('2/3','outer0'):RELEASE/'diagnostics'/'nu2of3_v4_outer_c3_p0.npz',
 ('2/3','outer1'):RELEASE/'diagnostics'/'nu2of3_v4_outer_c3_p1.npz',
 ('2/3','outer2'):RELEASE/'diagnostics'/'nu2of3_v4_outer_c3_p2.npz'}
BLUE='#28688C'; RED='#B5423A'; ORANGE='#E69F00'; MAGENTA='#CC79A7'
ED={'1/3':-37.3932329459257,'2/3':-52.72553897658505}
COLOR={'full':BLUE,'gamma':RED,'outer0':'#D55E00','outer1':'#7A5195','outer2':ORANGE}
LINESTYLE={'full':'-','gamma':'-','outer0':(0,(3,1.5)),'outer1':(0,(1.2,1.2)),'outer2':(0,(5,1.5))}


def selected_outer(filling):
 summary=json.loads((RELEASE/'summary.json').read_text())
 key='nu1of3' if filling=='1/3' else 'nu2of3'
 return f"outer{summary['selected_lowest_tail_sector'][key]}"


def style():
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
  'font.size':7.4,'axes.labelsize':7.5,'axes.titlesize':8,'axes.linewidth':.7,
  'xtick.labelsize':6.8,'ytick.labelsize':6.8,'xtick.major.size':3,'ytick.major.size':3,
  'xtick.direction':'out','ytick.direction':'out','pdf.fonttype':42,'figure.dpi':180,'savefig.dpi':400})


def ed_band_energies(filling):
 prefix,particles=('nu1of3',3) if filling=='1/3' else ('fig2',6)
 return np.asarray([float(np.load(DATA/f'{prefix}_{b}band_ground_state.npz')['energy_meV'])/particles for b in range(1,6)])


def main():
 style()
 fig,axs=plt.subplots(2,4,figsize=(7.25,4.45),gridspec_kw={'width_ratios':[1.18,1,1.05,1.12]})
 fig.subplots_adjust(left=.075,right=.99,bottom=.105,top=.84,wspace=.48,hspace=.43)
 for row,filling in enumerate(('1/3','2/3')):
  branches=('full','gamma',selected_outer(filling))
  traces={b:json.loads(RUNS[(filling,b)].read_text()) for b in branches}

  ax=axs[row,0]
  for b in branches:
   t=traces[b]; step=np.array([v['step'] for v in t]); en=np.array([v['energy_per_particle_meV'] for v in t]); acc=np.array([v['update_accepted'] for v in t])
   ax.plot(step,en,color=COLOR[b],ls=LINESTYLE[b],lw=1.0); ax.scatter(step[acc],en[acc],s=4,color=COLOR[b],alpha=.55,edgecolors='none')
  ax.axvline(20,color='.55',lw=.6,ls=(0,(2,2)),zorder=0)
  all_e=np.concatenate([[v['energy_per_particle_meV'] for v in t] for t in traces.values()])
  ax.set_xlim(0,122); ax.set_ylim(min(all_e.min(),ED[filling])-.15,-20 if filling=='1/3' else -40)
  ax.xaxis.set_major_locator(MaxNLocator(4,integer=True)); ax.yaxis.set_major_locator(MaxNLocator(4))
  ax.set_xlabel('NG update'); ax.set_ylabel(r'$E/N_e$ (meV)')

  ax=axs[row,1]; bands=np.arange(1,6); ed_curve=ed_band_energies(filling)
  ax.plot(bands,ed_curve,color='.22',lw=1.05,marker='o',markersize=4.7,markerfacecolor=MAGENTA,markeredgecolor='white',markeredgewidth=.5,zorder=3)
  energy_rows=[]
  for b,t in traces.items():
   tail=np.asarray([v['energy_per_particle_meV'] for v in t[-10:]])
   energy_rows.append((tail.mean(),tail.std(),COLOR[b]))
  for energy,sem,color in energy_rows:
   ax.axhline(energy,color=color,lw=.95,ls=(0,(4,2))); ax.axhspan(energy-sem,energy+sem,color=color,alpha=.08,lw=0)
  combined=np.r_[ed_curve,[r[0] for r in energy_rows]]; pad=max(.18,.13*np.ptp(combined))
  ax.set_xlim(.75,5.25); ax.set_ylim(float(combined.min()-pad),float(combined.max()+pad))
  ax.set_xticks([1,2,3,4,5]); ax.yaxis.set_major_locator(MaxNLocator(4)); ax.set_xlabel(r'retained bands $N_b$'); ax.set_ylabel(r'$E/N_e$ (meV)')

  ax=axs[row,2]; z_gamma=np.load(DIAG[(filling,'gamma')]); x=np.arange(1,6)
  bar_branches=branches; width=.19
  offsets={branches[0]:-1.5*width,branches[1]:-.5*width,branches[2]:.5*width}; ed_offset=1.5*width
  for b in bar_branches:
   z=np.load(DIAG[(filling,b)]); bw=100*z['band_weight']; sem=100*z['band_weight_sem']
   ax.bar(x+offsets[b],bw,width=width,color=COLOR[b],edgecolor='white',lw=.4,yerr=sem,error_kw={'elinewidth':.65,'capsize':1.3,'ecolor':COLOR[b]},zorder=3)
  ax.bar(x+ed_offset,100*z_gamma['five_band_ed_population'],width=width,color='#B9BEC1',edgecolor='white',lw=.4,zorder=2)
  ax.axhline(0,color='.25',lw=.55); ax.set_xlim(.5,5.5); ax.set_ylim(-6,105)
  ax.set_xticks(x); ax.set_yticks([0,50,100]); ax.set_xlabel('bare band index'); ax.set_ylabel('population (%)')

  ax=axs[row,3]
  for b in branches:
   t=traces[b]; step=np.array([v['step'] for v in t]); rr=np.array([v['relative_residual_norm'] for v in t])
   ax.plot(step,rr,color=COLOR[b],ls=LINESTYLE[b],lw=.85)
  ax.axvline(20,color='.55',lw=.6,ls=(0,(2,2)),zorder=0)
  ax.axhline(.02,color='.35',lw=.65,ls=(0,(2,2))); ax.axhline(.05,color='#D55E00',lw=.65,ls=(0,(3,2)))
  ax.set_yscale('log'); ax.set_xlim(0,122); ax.set_ylim(8e-3,max(.3,max(max(v['relative_residual_norm'] for v in t) for t in traces.values())*1.15))
  ax.xaxis.set_major_locator(MaxNLocator(4,integer=True)); ax.set_xlabel('NG update'); ax.set_ylabel(r'true $r_{\rm rel}$')

 for j,t in enumerate(('optimization','multiband energy','bare-band weight','CG stability')): axs[0,j].set_title(t,pad=4)
 handles=[Line2D([0],[0],color='.22',marker='o',mfc=MAGENTA,mec='white',lw=.9,label='multiband ED'),Line2D([0],[0],color=BLUE,lw=1,label=r'full $M$'),Line2D([0],[0],color=RED,lw=1,label=r'v4 $P_\Gamma[M]$'),Line2D([0],[0],color=ORANGE,lw=1,label=r'lowest-training v4 outer $P_m$'),Patch(facecolor='#B9BEC1',label='5-band ED weight')]
 fig.legend(handles=handles,frameon=False,loc='upper center',ncol=5,bbox_to_anchor=(.54,.995),columnspacing=.95,handletextpad=.38,fontsize=6.4)
 for lab,ax in zip('abcdefgh',axs.flat):
  ax.text(-.15,1.04,lab,transform=ax.transAxes,ha='left',va='bottom',fontsize=8.5,fontweight='bold')
  ax.grid(axis='y',color='.91',lw=.45,zorder=0); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
 FIG.mkdir(parents=True,exist_ok=True); fig.savefig(FIG/'fig6_neural_bloch_results.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig6_neural_bloch_results.png',bbox_inches='tight'); plt.close(fig)

def efficiency_timings():
 timing=json.loads((RELEASE/'summary.json').read_text())['timings_seconds']
 result={}
 for filling,key in [('1/3','nu1of3'),('2/3','nu2of3')]:
  t=timing[key]
  result[filling]=np.asarray([t['ed_5band_gamma'],t['full_m'],t['v4_gamma'],
                             t['v4_outer_c3_three_sector_wall']/3])
 return result


def make_training_summary():
 groups={'1/3':('outer0','outer1','outer2'),'2/3':('outer0','outer1','outer2')}
 fig,axs=plt.subplots(2,3,figsize=(8.4,5.15))
 fig.subplots_adjust(left=.065,right=.99,bottom=.12,top=.93,wspace=.43,hspace=.56)
 timing=efficiency_timings()
 for row,filling in enumerate(('1/3','2/3')):
  sector_traces=[json.loads(RUNS[(filling,b)].read_text()) for b in groups[filling]]
  ax=axs[row,0]; sector_colors=['#D55E00','#7A5195',ORANGE]
  zoom=ax.inset_axes([.36,.44,.62,.47])
  for m,t in enumerate(sector_traces):
   step=np.asarray([r['step'] for r in t]); en=np.asarray([r['energy_per_particle_meV'] for r in t])
   ax.plot(step,en,color=sector_colors[m],lw=1,label=rf'$m={m}$')
   zoom.plot(step[step>=21],en[step>=21],color=sector_colors[m],lw=.8)
  ax.axvline(20,color='.55',ls=':',lw=.7)
  ax.set_xlim(0,122); ax.set_xlabel('NG update'); ax.set_ylabel(r'$E/N_e$ (meV)')
  ax.set_title(rf'$\nu={filling}$: optimization',fontweight='bold')
  ax.legend(loc='center right',bbox_to_anchor=(1,.28),fontsize=5.8,ncol=3,frameon=False,handlelength=.9,columnspacing=.5)
  zoom.set_xlim(20,121); zoom.set_xticks([20,70,120]); zoom.tick_params(labelsize=5.5,length=2,pad=1)
  zoom.yaxis.set_major_locator(MaxNLocator(3)); zoom.set_title('updates 21--120',fontsize=5.8,pad=2)
  zoom.spines['top'].set_visible(False); zoom.spines['right'].set_visible(False)
  means=np.asarray([np.mean([v['energy_per_particle_meV'] for v in t[-10:]]) for t in sector_traces])
  rms=np.asarray([np.std([v['energy_per_particle_meV'] for v in t[-10:]]) for t in sector_traces]); delta=means-means.min()
  ax=axs[row,1]; x=np.arange(3)
  bars=ax.bar(x,delta,yerr=rms,color=sector_colors,width=.60,edgecolor='white',error_kw={'elinewidth':.75,'capsize':2,'capthick':.75})
  ax.bar_label(bars,labels=[f'{v:.2f}' for v in delta],padding=2,fontsize=6.8); ax.axhline(0,color='.25',lw=.6)
  ax.set_ylim(-.12,max(delta+rms)*1.18+.08); ax.set_xticks(x,[r'$m=0$',r'$m=1$',r'$m=2$'])
  ax.set_ylabel(r'$\Delta E_{\rm tail}/N_e$ (meV)'); ax.set_title(rf'$\nu={filling}$: sector offsets',fontweight='bold')

  ax=axs[row,2]; seconds=timing[filling]
  bars=ax.bar(np.arange(4),seconds-.1,bottom=.1,color=['#7A7F83',BLUE,RED,ORANGE],
              width=.60,edgecolor='white',zorder=3)
  bars[-1].set_hatch('///')
  for bar,value in zip(bars,seconds):
   label=f'{value:.2f} s' if value<60 else f'{value/60:.1f} min'
   ax.annotate(label,(bar.get_x()+bar.get_width()/2,value),xytext=(0,3),textcoords='offset points',
               ha='center',va='bottom',fontsize=6)
  ax.set_xticks(np.arange(4),[r'ED'+'\n'+r'$5b,\ \Gamma$',r'full $M$',r'v4 $P_\Gamma$'+'\n'+r'$[M]$','v4 Outer\nper sector'])
  ax.tick_params(axis='x',labelsize=6)
  ax.set_yscale('log'); ax.set_ylim(.1,6000)
  ax.set_yticks([.1,1,10,100,1000],['0.1','1','10','100','1000'])
  ax.minorticks_off(); ax.set_ylabel('wall time (s, log scale)')
  ax.set_title(rf'$\nu={filling}$: 9-cell timing',fontweight='bold')

 for label,ax in zip('abcdef',axs.flat):
  ax.text(-.14,1.04,label,transform=ax.transAxes,fontsize=8.5,fontweight='bold')
  ax.grid(axis='y',color='.92',lw=.45,zorder=0); ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False); ax.tick_params(length=2.8)
 FIG.mkdir(parents=True,exist_ok=True)
 fig.savefig(FIG/'fig6_training_summary.pdf',bbox_inches='tight')
 fig.savefig(FIG/'fig6_training_summary.png',bbox_inches='tight'); plt.close(fig)

if __name__=='__main__': main(); make_training_summary()
