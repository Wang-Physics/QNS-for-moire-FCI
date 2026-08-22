"""Final publication-style 120-step optimization, validation, band-weight and CG figure."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'result'/'data'; FIG=ROOT/'result'/'figures'
RELEASE=ROOT/'result'/'release_data'; VAL=RELEASE
RUNS={
 ('1/3','full'):RELEASE/'traces'/'nu1of3_full_m.json',
 ('1/3','gamma'):RELEASE/'traces'/'nu1of3_no_m_gamma.json',
 ('2/3','full'):RELEASE/'traces'/'nu2of3_full_m.json',
 ('2/3','gamma'):RELEASE/'traces'/'nu2of3_no_m_gamma.json'}
DIAG={
 ('1/3','full'):RELEASE/'diagnostics'/'nu1of3_full_m.npz',
 ('1/3','gamma'):RELEASE/'diagnostics'/'nu1of3_no_m_gamma.npz',
 ('2/3','full'):RELEASE/'diagnostics'/'nu2of3_full_m.npz',
 ('2/3','gamma'):RELEASE/'diagnostics'/'nu2of3_no_m_gamma.npz'}
BLUE='#28688C'; RED='#B5423A'; MAGENTA='#CC79A7'; GRAY='#777A7C'; ED={'1/3':-37.3932329459257,'2/3':-52.72553897658505}
LABEL={'full':r'full $M$','gamma':r'no $M$, $\Gamma$'}

def style():
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
  'font.size':7.4,'axes.labelsize':7.5,'axes.titlesize':8,'axes.linewidth':.7,
  'xtick.labelsize':6.8,'ytick.labelsize':6.8,'xtick.major.size':3,'ytick.major.size':3,
  'xtick.direction':'out','ytick.direction':'out','pdf.fonttype':42,'figure.dpi':180,'savefig.dpi':400})

def ed_band_energies(filling):
 prefix,particles=('nu1of3',3) if filling=='1/3' else ('fig2',6)
 return np.asarray([float(np.load(DATA/f'{prefix}_{b}band_ground_state.npz')['energy_meV'])/particles for b in range(1,6)])

def main():
 style(); validation=json.loads((VAL/'validation_summary.json').read_text())
 fig,axs=plt.subplots(2,4,figsize=(7.25,4.45),gridspec_kw={'width_ratios':[1.18,1,1.05,1.12]})
 fig.subplots_adjust(left=.085,right=.99,bottom=.105,top=.84,wspace=.48,hspace=.43)
 for row,filling in enumerate(('1/3','2/3')):
  traces={b:json.loads(RUNS[(filling,b)].read_text()) for b in ('full','gamma')}
  ax=axs[row,0]
  for b,c in [('full',BLUE),('gamma',RED)]:
   t=traces[b]; step=np.array([v['step'] for v in t]); en=np.array([v['energy_per_particle_meV'] for v in t]); acc=np.array([v['update_accepted'] for v in t])
   ax.plot(step,en,color=c,lw=1.0); ax.scatter(step[acc],en[acc],s=4,color=c,alpha=.55,edgecolors='none')
   ax.scatter(step[~acc],en[~acc],s=16,marker='x',color='#222222',lw=.7,zorder=4)
  ax.axhline(ED[filling],color='.25',lw=.8,ls=(0,(2,2)))
  fk='nu1of3' if filling=='1/3' else 'nu2of3'
  precise_key=fk+'_no_m_gamma'
  precise=validation['precise'][precise_key]
  precise_energy=float(precise['energy_per_particle_meV'])
  precise_sem=float(precise['blocked_sem_per_particle_meV'])
  ax.axhline(precise_energy,color=RED,lw=.9,ls=(0,(4,2)))
  ax.axhspan(precise_energy-precise_sem,precise_energy+precise_sem,color=RED,alpha=.08,lw=0)
  all_e=np.concatenate([[v['energy_per_particle_meV'] for v in t] for t in traces.values()])
  ax.set_xlim(0,122); ax.set_ylim(min(all_e.min(),ED[filling],precise_energy)-.15,-20 if filling=='1/3' else -40)
  ax.xaxis.set_major_locator(MaxNLocator(4,integer=True)); ax.yaxis.set_major_locator(MaxNLocator(4))
  ax.set_xlabel('NG update'); ax.set_ylabel(r'$E/N_e$ (meV)')

  ax=axs[row,1]
  bands=np.arange(1,6); ed_curve=ed_band_energies(filling)
  low=validation['low_cost']; full_key=fk+'_full_m'
  full_energy=float(low[full_key]['energy_per_particle_meV'])
  full_sem=float(low[full_key]['blocked_sem_per_particle_meV'])
  ax.plot(bands,ed_curve,color='.22',lw=1.05,marker='o',markersize=4.7,
   markerfacecolor=MAGENTA,markeredgecolor='white',markeredgewidth=.5,zorder=3)
  ax.axhline(full_energy,color=BLUE,lw=.95,ls=(0,(4,2)))
  ax.axhspan(full_energy-full_sem,full_energy+full_sem,color=BLUE,alpha=.07,lw=0)
  ax.axhline(precise_energy,color=RED,lw=.95,ls=(0,(4,2)))
  ax.axhspan(precise_energy-precise_sem,precise_energy+precise_sem,color=RED,alpha=.08,lw=0)
  combined=np.r_[ed_curve,full_energy,precise_energy]; pad=max(.18,.13*np.ptp(combined))
  ax.set_xlim(.75,5.25); ax.set_ylim(float(combined.min()-pad),float(combined.max()+pad))
  ax.set_xticks([1,2,3,4,5]); ax.yaxis.set_major_locator(MaxNLocator(4))
  ax.set_xlabel(r'retained bands $N_b$'); ax.set_ylabel(r'$E/N_e$ (meV)')

  ax=axs[row,2]
  z_full=np.load(DIAG[(filling,'full')]); z_gamma=np.load(DIAG[(filling,'gamma')])
  x=np.arange(1,6); w=.24
  full_bw=100*z_full['band_weight']; full_sem=100*z_full['band_weight_sem']
  gamma_bw=100*z_gamma['band_weight']; gamma_sem=100*z_gamma['band_weight_sem']
  ed_bw=100*z_gamma['five_band_ed_population']
  ax.bar(x-w,full_bw,width=w,color=BLUE,edgecolor='white',lw=.4,yerr=full_sem,
   error_kw={'elinewidth':.65,'capsize':1.5,'ecolor':'#20516C'},zorder=3)
  ax.bar(x,gamma_bw,width=w,color=RED,edgecolor='white',lw=.4,yerr=gamma_sem,
   error_kw={'elinewidth':.65,'capsize':1.5,'ecolor':'#7E2D29'},zorder=3)
  ax.bar(x+w,ed_bw,width=w,color='#B9BEC1',edgecolor='white',lw=.4,zorder=2)
  ax.axhline(0,color='.25',lw=.55); ax.set_xlim(.5,5.5); ax.set_ylim(-6,105)
  ax.set_xticks(x); ax.set_yticks([0,50,100]); ax.set_xlabel('bare band index'); ax.set_ylabel('population (%)')

  ax=axs[row,3]
  for b,c in [('full',BLUE),('gamma',RED)]:
   t=traces[b]; step=np.array([v['step'] for v in t]); rr=np.array([v['relative_residual_norm'] for v in t]); acc=np.array([v['update_accepted'] for v in t])
   ax.plot(step,rr,color=c,lw=.85); ax.scatter(step[~acc],rr[~acc],marker='x',s=17,color='#222',lw=.7,zorder=4)
  ax.axhline(.02,color='.35',lw=.65,ls=(0,(2,2))); ax.axhline(.05,color='#D55E00',lw=.65,ls=(0,(3,2)))
  ax.set_yscale('log'); ax.set_xlim(0,122); ax.set_ylim(8e-3,max(.3,max(max(v['relative_residual_norm'] for v in t) for t in traces.values())*1.15))
  ax.xaxis.set_major_locator(MaxNLocator(4,integer=True)); ax.set_xlabel('NG update'); ax.set_ylabel(r'true $r_{\rm rel}$')

 for j,t in enumerate(('optimization','multiband energy','bare-band weight','CG stability')): axs[0,j].set_title(t,pad=4)
 fig.text(.018,.61,r'$\nu=1/3$',rotation=90,ha='center',va='center',fontsize=8.5,fontweight='bold')
 fig.text(.018,.22,r'$\nu=2/3$',rotation=90,ha='center',va='center',fontsize=8.5,fontweight='bold')
 handles=[Line2D([0],[0],color='.22',marker='o',mfc=MAGENTA,mec='white',lw=.9,label='multiband ED'),Line2D([0],[0],color=BLUE,ls=(0,(4,2)),lw=.9,label=r'full $M$ validation'),Line2D([0],[0],color=RED,ls=(0,(4,2)),lw=.9,label=r'no $M$ precise validation'),Patch(facecolor=BLUE,label=r'full $M$ band weight'),Patch(facecolor=RED,label=r'no $M$ band weight'),Patch(facecolor='#B9BEC1',label='5-band ED weight')]
 fig.legend(handles=handles,frameon=False,loc='upper center',ncol=6,bbox_to_anchor=(.54,.995),columnspacing=.9,handletextpad=.4,fontsize=6.6)
 for lab,ax in zip('abcdefgh',axs.flat):
  ax.text(-.15,1.04,lab,transform=ax.transAxes,ha='left',va='bottom',fontsize=8.5,fontweight='bold')
  ax.grid(axis='y',color='.91',lw=.45,zorder=0); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
 FIG.mkdir(parents=True,exist_ok=True); fig.savefig(FIG/'fig6_neural_bloch_results.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig6_neural_bloch_results.png',bbox_inches='tight'); plt.close(fig)

if __name__=='__main__': main()
