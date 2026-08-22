"""Four-row full-M/no-M observables from independent step-120 validation ensembles."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .run_neural_bloch import neural_bloch_inputs

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'result'/'release_data'/'diagnostics'
FIG=ROOT/'result'/'figures'
FILES={
 ('1/3','full'):DATA/'nu1of3_full_m.npz',
 ('1/3','gamma'):DATA/'nu1of3_no_m_gamma.npz',
 ('2/3','full'):DATA/'nu2of3_full_m.npz',
 ('2/3','gamma'):DATA/'nu2of3_no_m_gamma.npz',
}
BLUE='#28688C'; RED='#B5423A'

def style():
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
  'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'axes.linewidth':.7,
  'xtick.labelsize':7,'ytick.labelsize':7,'xtick.direction':'out','ytick.direction':'out',
  'pdf.fonttype':42,'figure.dpi':180,'savefig.dpi':400})

def hexagon(ax,model):
 v=np.asarray([(2*model.b1+model.b2)/3,(model.b1+2*model.b2)/3,
  (-model.b1+model.b2)/3,-(2*model.b1+model.b2)/3,
  -(model.b1+2*model.b2)/3,-(-model.b1+model.b2)/3])
 c=np.vstack([v,v[0]])
 ax.plot(c[:,0],c[:,1],color='#42484C',lw=.78,zorder=1,clip_on=False)
 # Leave a marker-radius margin around the complete hexagonal BZ.
 ax.set_xlim(-.76,.76); ax.set_ylim(-.73,.73); ax.set_aspect('equal')
 ax.set_xticks([]); ax.set_yticks([])
 for spine in ax.spines.values(): spine.set_visible(False)

def main():
 style(); model,_=neural_bloch_inputs()
 loaded={key:np.load(path) for key,path in FILES.items()}
 rows=[('1/3','full',r'full $M$'),('1/3','gamma',r'no $M$, $\Gamma$'),
       ('2/3','full',r'full $M$'),('2/3','gamma',r'no $M$, $\Gamma$')]
 smax=max(float(loaded[(f,b)]['charge_structure_factor_full'].max()) for f,b,_ in rows)
 rhoall=np.concatenate([loaded[(f,b)]['charge_density_over_mean'].ravel() for f,b,_ in rows])
 rmin,rmax=np.percentile(rhoall,[1,99])
 fig,axs=plt.subplots(4,3,figsize=(7.25,8.15),gridspec_kw={'width_ratios':[1,1,1.22]})
 fig.subplots_adjust(left=.125,right=.94,bottom=.055,top=.95,wspace=.38,hspace=.25)
 arts=[None,None,None]
 for row,(filling,branch,method) in enumerate(rows):
  z=loaded[(filling,branch)]
  kp=np.asarray([model.wrap_to_hexagon(k) for k in z['momentum_k_points']])
  ax=axs[row,0]; hexagon(ax,model)
  arts[0]=ax.scatter(kp[:,0],kp[:,1],c=z['momentum_occupation_band1'],s=145,
   cmap='viridis',vmin=0,vmax=1,edgecolors='white',linewidths=.62,zorder=3,clip_on=False)
  ax=axs[row,1]; hexagon(ax,model)
  qp=np.asarray([model.wrap_to_hexagon(q) for q in z['structure_q_vectors']])
  arts[1]=ax.scatter(qp[:,0],qp[:,1],c=z['charge_structure_factor_full'],s=145,
   cmap='magma',vmin=0,vmax=smax,edgecolors='white',linewidths=.62,zorder=3,clip_on=False)
  ax=axs[row,2]
  x=np.asarray(z['density_x_fraction']); y=np.asarray(z['density_y_fraction'])
  density=np.tile(np.asarray(z['charge_density_over_mean']),(2,2))
  tx=np.concatenate([x,x+1]); ty=np.concatenate([y,y+1])
  fx,fy=np.meshgrid(tx,ty,indexing='ij'); cx=fx+.5*fy; cy=np.sqrt(3)*fy/2
  arts[2]=ax.pcolormesh(cx,cy,density,shading='nearest',cmap='magma',
   vmin=rmin,vmax=rmax,rasterized=True,clip_on=False)
  ax.plot([0,2,3,1,0],[0,0,np.sqrt(3),np.sqrt(3),0],color='#34383B',lw=.75,clip_on=False)
  # The old limits clipped the half-pixel edges of the periodic two-by-two map.
  ax.set_aspect('equal'); ax.set_xlim(-.14,3.14); ax.set_ylim(-.12,np.sqrt(3)+.12)
  ax.set_xticks([0,1.5,3]); ax.set_yticks([0,np.sqrt(3)]); ax.set_yticklabels(['0',r'$\sqrt{3}$'])
  ax.set_xlabel(r'$x/a_M$'); ax.set_ylabel(r'$y/a_M$')
  ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
  axs[row,0].text(-.37,.5,rf'$\nu={filling}$'+'\n'+method,
   transform=axs[row,0].transAxes,rotation=90,ha='center',va='center',fontsize=8,fontweight='bold')
 for j,title in enumerate((r'$n_1(\mathbf{k})$',r'$S_{\rm full}(\mathbf{q})$',r'$\rho(\mathbf{r})/\bar\rho$')):
  axs[0,j].set_title(title,pad=5)
 for label,ax in zip('abcdefghijkl',axs.flat):
  ax.text(-.11,1.03,label,transform=ax.transAxes,ha='left',va='bottom',fontsize=9,fontweight='bold')
 c0=fig.colorbar(arts[0],ax=axs[:,0].tolist(),fraction=.025,pad=.02,aspect=36)
 c0.set_ticks([0,.5,1]); c0.ax.tick_params(length=2)
 c1=fig.colorbar(arts[1],ax=axs[:,1].tolist(),fraction=.025,pad=.02,aspect=36)
 c1.set_ticks([0,smax/2,smax]); c1.ax.tick_params(length=2)
 c2=fig.colorbar(arts[2],ax=axs[:,2].tolist(),fraction=.025,pad=.02,aspect=36)
 c2.set_ticks([round(rmin,1),1,round(rmax,1)]); c2.ax.tick_params(length=2)
 FIG.mkdir(parents=True,exist_ok=True)
 fig.savefig(FIG/'fig7_neural_bloch_observables.pdf',bbox_inches='tight',pad_inches=.08)
 fig.savefig(FIG/'fig7_neural_bloch_observables.png',bbox_inches='tight',pad_inches=.08)
 plt.close(fig)

if __name__=='__main__': main()
