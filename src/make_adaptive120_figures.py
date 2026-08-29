"""Publication-style optimization, validation, band-weight and CG figure."""
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
 ('1/3','c3'):RELEASE/'traces'/'nu1of3_c3_qns.json',
 ('1/3','outer'):RELEASE/'traces'/'nu1of3_outer_c3_p0.json',
 ('2/3','full'):RELEASE/'traces'/'nu2of3_full_m.json',
 ('2/3','gamma'):RELEASE/'traces'/'nu2of3_no_m_gamma.json',
 ('2/3','outer0'):RELEASE/'traces'/'nu2of3_outer_c3_p0.json',
 ('2/3','outer1'):RELEASE/'traces'/'nu2of3_outer_c3_p1.json',
 ('2/3','outer2'):RELEASE/'traces'/'nu2of3_outer_c3_p2.json'}
DIAG={
 ('1/3','full'):RELEASE/'diagnostics'/'nu1of3_full_m.npz',
 ('1/3','gamma'):RELEASE/'diagnostics'/'nu1of3_no_m_gamma.npz',
 ('1/3','c3'):RELEASE/'diagnostics'/'nu1of3_c3_qns.npz',
 ('1/3','outer'):RELEASE/'diagnostics'/'nu1of3_outer_c3_p0.npz',
 ('2/3','full'):RELEASE/'diagnostics'/'nu2of3_full_m.npz',
 ('2/3','gamma'):RELEASE/'diagnostics'/'nu2of3_no_m_gamma.npz',
 ('2/3','outer0'):RELEASE/'diagnostics'/'nu2of3_outer_c3_p0.npz'}
BLUE='#28688C'; RED='#B5423A'; GREEN='#009E73'; ORANGE='#E69F00'; MAGENTA='#CC79A7'
ED={'1/3':-37.3932329459257,'2/3':-52.72553897658505}
COLOR={'full':BLUE,'gamma':RED,'c3':GREEN,'outer':ORANGE,'outer0':'#D55E00','outer1':'#7A5195','outer2':ORANGE}
LINESTYLE={'full':'-','gamma':'-','c3':'-','outer':'-','outer0':(0,(3,1.5)),'outer1':(0,(1.2,1.2)),'outer2':(0,(5,1.5))}


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
 outer_validation=json.loads((RELEASE/'outer_c3_validation_summary.json').read_text())
 outer_nu2=json.loads((RELEASE/'outer_c3_nu2of3_validation_summary.json').read_text())
 fig,axs=plt.subplots(2,4,figsize=(7.25,4.45),gridspec_kw={'width_ratios':[1.18,1,1.05,1.12]})
 fig.subplots_adjust(left=.085,right=.99,bottom=.105,top=.84,wspace=.48,hspace=.43)
 for row,filling in enumerate(('1/3','2/3')):
  branches=('full','gamma','c3','outer') if filling=='1/3' else ('full','gamma','outer0','outer1','outer2')
  traces={b:json.loads(RUNS[(filling,b)].read_text()) for b in branches}
  fk='nu1of3' if filling=='1/3' else 'nu2of3'
  precise=validation['precise'][fk+'_no_m_gamma']
  precise_energy=float(precise['energy_per_particle_meV']); precise_sem=float(precise['blocked_sem_per_particle_meV'])
  if filling=='1/3':
   c3v=validation['precise']['nu1of3_c3_qns']
   c3e=float(c3v['energy_per_particle_meV']); c3s=float(c3v['blocked_sem_per_particle_meV'])
   outerv=outer_validation['precise']['m0']
   outere=float(outerv['energy_per_particle_meV']); outers=float(outerv['blocked_sem_per_particle_meV'])
  else:
   outer_low=[outer_nu2['low_cost'][f'p{m}'] for m in range(3)]
   outer_precise=outer_nu2['precise']['p0']

  ax=axs[row,0]
  for b in branches:
   t=traces[b]; step=np.array([v['step'] for v in t]); en=np.array([v['energy_per_particle_meV'] for v in t]); acc=np.array([v['update_accepted'] for v in t])
   ax.plot(step,en,color=COLOR[b],ls=LINESTYLE[b],lw=1.0); ax.scatter(step[acc],en[acc],s=4,color=COLOR[b],alpha=.55,edgecolors='none')
   ax.scatter(step[~acc],en[~acc],s=16,marker='x',color='#222222',lw=.7,zorder=4)
  all_e=np.concatenate([[v['energy_per_particle_meV'] for v in t] for t in traces.values()])
  compare=[ED[filling],precise_energy]+([c3e,outere] if filling=='1/3' else [float(v['energy_per_particle_meV']) for v in outer_low]+[float(outer_precise['energy_per_particle_meV'])])
  ax.set_xlim(0,122); ax.set_ylim(min(all_e.min(),*compare)-.15,-20 if filling=='1/3' else -40)
  ax.xaxis.set_major_locator(MaxNLocator(4,integer=True)); ax.yaxis.set_major_locator(MaxNLocator(4))
  ax.set_xlabel('NG update'); ax.set_ylabel(r'$E/N_e$ (meV)')

  ax=axs[row,1]; bands=np.arange(1,6); ed_curve=ed_band_energies(filling)
  full=validation['low_cost'][fk+'_full_m']; full_energy=float(full['energy_per_particle_meV']); full_sem=float(full['blocked_sem_per_particle_meV'])
  ax.plot(bands,ed_curve,color='.22',lw=1.05,marker='o',markersize=4.7,markerfacecolor=MAGENTA,markeredgecolor='white',markeredgewidth=.5,zorder=3)
  energy_rows=[(full_energy,full_sem,BLUE),(precise_energy,precise_sem,RED)]
  if filling=='1/3':
   energy_rows += [(c3e,c3s,GREEN),(outere,outers,ORANGE)]
  else:
   energy_rows += [(float(outer_precise['energy_per_particle_meV']),float(outer_precise['blocked_sem_per_particle_meV']),COLOR['outer0']),
                   (float(outer_low[1]['energy_per_particle_meV']),float(outer_low[1]['blocked_sem_per_particle_meV']),COLOR['outer1']),
                   (float(outer_low[2]['energy_per_particle_meV']),float(outer_low[2]['blocked_sem_per_particle_meV']),COLOR['outer2'])]
  for energy,sem,color in energy_rows:
   ax.axhline(energy,color=color,lw=.95,ls=(0,(4,2))); ax.axhspan(energy-sem,energy+sem,color=color,alpha=.08,lw=0)
  combined=np.r_[ed_curve,[r[0] for r in energy_rows]]; pad=max(.18,.13*np.ptp(combined))
  ax.set_xlim(.75,5.25); ax.set_ylim(float(combined.min()-pad),float(combined.max()+pad))
  ax.set_xticks([1,2,3,4,5]); ax.yaxis.set_major_locator(MaxNLocator(4)); ax.set_xlabel(r'retained bands $N_b$'); ax.set_ylabel(r'$E/N_e$ (meV)')

  ax=axs[row,2]; z_gamma=np.load(DIAG[(filling,'gamma')]); x=np.arange(1,6)
  if filling=='1/3':
   bar_branches=('full','gamma','c3','outer'); width=.15
   offsets={'full':-2*width,'gamma':-width,'c3':0,'outer':width}; ed_offset=2*width
  else:
   bar_branches=('full','gamma','outer0'); width=.19
   offsets={'full':-1.5*width,'gamma':-.5*width,'outer0':.5*width}; ed_offset=1.5*width
  for b in bar_branches:
   z=np.load(DIAG[(filling,b)]); bw=100*z['band_weight']; sem=100*z['band_weight_sem']
   ax.bar(x+offsets[b],bw,width=width,color=COLOR[b],edgecolor='white',lw=.4,yerr=sem,error_kw={'elinewidth':.65,'capsize':1.3,'ecolor':COLOR[b]},zorder=3)
  ax.bar(x+ed_offset,100*z_gamma['five_band_ed_population'],width=width,color='#B9BEC1',edgecolor='white',lw=.4,zorder=2)
  ax.axhline(0,color='.25',lw=.55); ax.set_xlim(.5,5.5); ax.set_ylim(-6,105)
  ax.set_xticks(x); ax.set_yticks([0,50,100]); ax.set_xlabel('bare band index'); ax.set_ylabel('population (%)')

  ax=axs[row,3]
  for b in branches:
   t=traces[b]; step=np.array([v['step'] for v in t]); rr=np.array([v['relative_residual_norm'] for v in t]); acc=np.array([v['update_accepted'] for v in t])
   ax.plot(step,rr,color=COLOR[b],ls=LINESTYLE[b],lw=.85); ax.scatter(step[~acc],rr[~acc],marker='x',s=17,color='#222',lw=.7,zorder=4)
  ax.axhline(.02,color='.35',lw=.65,ls=(0,(2,2))); ax.axhline(.05,color='#D55E00',lw=.65,ls=(0,(3,2)))
  ax.set_yscale('log'); ax.set_xlim(0,122); ax.set_ylim(8e-3,max(.3,max(max(v['relative_residual_norm'] for v in t) for t in traces.values())*1.15))
  ax.xaxis.set_major_locator(MaxNLocator(4,integer=True)); ax.set_xlabel('NG update'); ax.set_ylabel(r'true $r_{\rm rel}$')

 for j,t in enumerate(('optimization','multiband energy','bare-band weight','CG stability')): axs[0,j].set_title(t,pad=4)
 fig.text(.018,.61,r'$\nu=1/3$',rotation=90,ha='center',va='center',fontsize=8.5,fontweight='bold')
 fig.text(.018,.22,r'$\nu=2/3$',rotation=90,ha='center',va='center',fontsize=8.5,fontweight='bold')
 handles=[Line2D([0],[0],color='.22',marker='o',mfc=MAGENTA,mec='white',lw=.9,label='multiband ED'),Line2D([0],[0],color=BLUE,lw=1,label=r'full $M$'),Line2D([0],[0],color=RED,lw=1,label=r'no $M$'),Line2D([0],[0],color=GREEN,lw=1,label='internal C3'),Line2D([0],[0],color=COLOR['outer0'],ls=LINESTYLE['outer0'],lw=1,label=r'outer $P_0$'),Line2D([0],[0],color=COLOR['outer1'],ls=LINESTYLE['outer1'],lw=1,label=r'outer $P_1$'),Line2D([0],[0],color=COLOR['outer2'],ls=LINESTYLE['outer2'],lw=1,label=r'outer $P_2$'),Patch(facecolor='#B9BEC1',label='5-band ED weight')]
 fig.legend(handles=handles,frameon=False,loc='upper center',ncol=8,bbox_to_anchor=(.54,.995),columnspacing=.72,handletextpad=.35,fontsize=6.15)
 for lab,ax in zip('abcdefgh',axs.flat):
  ax.text(-.15,1.04,lab,transform=ax.transAxes,ha='left',va='bottom',fontsize=8.5,fontweight='bold')
  ax.grid(axis='y',color='.91',lw=.45,zorder=0); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
 FIG.mkdir(parents=True,exist_ok=True); fig.savefig(FIG/'fig6_neural_bloch_results.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig6_neural_bloch_results.png',bbox_inches='tight'); plt.close(fig)

if __name__=='__main__': main()
