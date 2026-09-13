"""Filling-resolved outer-C3 sector observables from training ensembles."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .run_neural_bloch import neural_bloch_inputs

ROOT=Path(__file__).resolve().parents[1]
REPORT_DATA=ROOT/'result'/'data'/'local_v4_final'
DATA=REPORT_DATA/'diagnostics'; FIG=ROOT/'result'/'figures'
FILES={
 ('1/3','full'):DATA/'nu1of3_full_m.npz',
 ('1/3','gamma'):DATA/'nu1of3_v4_gamma.npz',
 ('1/3','outer0'):DATA/'nu1of3_v4_outer_c3_p0.npz',
 ('1/3','outer1'):DATA/'nu1of3_v4_outer_c3_p1.npz',
 ('1/3','outer2'):DATA/'nu1of3_v4_outer_c3_p2.npz',
 ('2/3','full'):DATA/'nu2of3_full_m.npz',
 ('2/3','gamma'):DATA/'nu2of3_v4_gamma.npz',
 ('2/3','outer0'):DATA/'nu2of3_v4_outer_c3_p0.npz',
 ('2/3','outer1'):DATA/'nu2of3_v4_outer_c3_p1.npz',
 ('2/3','outer2'):DATA/'nu2of3_v4_outer_c3_p2.npz'}


def selected_sectors():
 summary=__import__('json').loads((REPORT_DATA/'summary.json').read_text())
 return {'1/3':int(summary['selected_lowest_tail_sector']['nu1of3']),
         '2/3':int(summary['selected_lowest_tail_sector']['nu2of3'])}


def style():
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
  'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'axes.linewidth':.7,
  'xtick.labelsize':7,'ytick.labelsize':7,'xtick.direction':'out','ytick.direction':'out',
  'pdf.fonttype':42,'figure.dpi':180,'savefig.dpi':400})


def hexagon(ax,model):
 v=np.asarray([(2*model.b1+model.b2)/3,(model.b1+2*model.b2)/3,
  (-model.b1+model.b2)/3,-(2*model.b1+model.b2)/3,
  -(model.b1+2*model.b2)/3,-(-model.b1+model.b2)/3])
 c=np.vstack([v,v[0]]); ax.plot(c[:,0],c[:,1],color='#42484C',lw=.78,zorder=1,clip_on=False)
 ax.set_xlim(-.76,.76); ax.set_ylim(-.73,.73); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
 for spine in ax.spines.values(): spine.set_visible(False)


def make_combined():
 style(); model,_=neural_bloch_inputs(); loaded={key:np.load(path) for key,path in FILES.items()}
 rows=[('1/3','full',r'full $M$'),
       ('1/3','gamma',r'v4 $P_\Gamma[M]$'),
       ('2/3','full',r'full $M$'),
       ('2/3','gamma',r'v4 $P_\Gamma[M]$')]
 smax=max(float(loaded[(f,b)]['charge_structure_factor_full'].max()) for f,b,_ in rows)
 rhoall=np.concatenate([loaded[(f,b)]['charge_density_over_mean'].ravel() for f,b,_ in rows]); rmin,rmax=np.percentile(rhoall,[1,99])
 fig,axs=plt.subplots(len(rows),3,figsize=(7.25,8.0),gridspec_kw={'width_ratios':[1,1,1.22]})
 fig.subplots_adjust(left=.125,right=.94,bottom=.068,top=.965,wspace=.38,hspace=.22)
 arts=[None,None,None]
 for row,(filling,branch,method) in enumerate(rows):
  z=loaded[(filling,branch)]; kp=np.asarray([model.wrap_to_hexagon(k) for k in z['momentum_k_points']])
  ax=axs[row,0]; hexagon(ax,model)
  arts[0]=ax.scatter(kp[:,0],kp[:,1],c=z['momentum_occupation_band1'],s=145,cmap='viridis',vmin=0,vmax=1,edgecolors='white',linewidths=.62,zorder=3,clip_on=False)
  ax=axs[row,1]; hexagon(ax,model); qp=np.asarray([model.wrap_to_hexagon(q) for q in z['structure_q_vectors']])
  arts[1]=ax.scatter(qp[:,0],qp[:,1],c=z['charge_structure_factor_full'],s=145,cmap='magma',vmin=0,vmax=smax,edgecolors='white',linewidths=.62,zorder=3,clip_on=False)
  ax=axs[row,2]; x=np.asarray(z['density_x_fraction']); y=np.asarray(z['density_y_fraction'])
  density=np.tile(np.asarray(z['charge_density_over_mean']),(2,2)); tx=np.concatenate([x,x+1]); ty=np.concatenate([y,y+1])
  fx,fy=np.meshgrid(tx,ty,indexing='ij'); cx=fx+.5*fy; cy=np.sqrt(3)*fy/2
  arts[2]=ax.pcolormesh(cx,cy,density,shading='nearest',cmap='magma',vmin=rmin,vmax=rmax,rasterized=True,clip_on=False)
  ax.plot([0,2,3,1,0],[0,0,np.sqrt(3),np.sqrt(3),0],color='#34383B',lw=.75,clip_on=False)
  ax.set_aspect('equal'); ax.set_xlim(-.14,3.14); ax.set_ylim(-.12,np.sqrt(3)+.12)
  ax.set_xticks([0,1.5,3]); ax.set_yticks([0,np.sqrt(3)]); ax.set_yticklabels(['0',r'$\sqrt{3}$'])
  ax.set_xlabel(r'$x/a_M$'); ax.set_ylabel(r'$y/a_M$'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
  axs[row,0].text(-.37,.5,rf'$\nu={filling}$'+'\n'+method,transform=axs[row,0].transAxes,rotation=90,ha='center',va='center',fontsize=8,fontweight='bold')
 for j,title in enumerate((r'$n_1(\mathbf{k})$',r'$S_{\rm full}(\mathbf{q})$',r'$\rho(\mathbf{r})/\bar\rho$')): axs[0,j].set_title(title,pad=5)
 for label,ax in zip('abcdefghijklmnopqr',axs.flat): ax.text(-.11,1.03,label,transform=ax.transAxes,ha='left',va='bottom',fontsize=9,fontweight='bold')
 cbars=[]
 for j,(art,ticks) in enumerate(((arts[0],[0,.5,1]),(arts[1],[0,smax/2,smax]),(arts[2],[round(rmin,1),1,round(rmax,1)]))):
  pos=axs[-1,j].get_position(); cax=fig.add_axes([pos.x0,.022,pos.width,.009])
  cb=fig.colorbar(art,cax=cax,orientation='horizontal'); cb.set_ticks(ticks); cb.ax.tick_params(length=2,pad=1)
  cb.outline.set_linewidth(.55); cbars.append(cb)
 FIG.mkdir(parents=True,exist_ok=True); fig.savefig(FIG/'fig7_neural_bloch_observables.pdf',bbox_inches='tight',pad_inches=.08); fig.savefig(FIG/'fig7_neural_bloch_observables.png',bbox_inches='tight',pad_inches=.08); plt.close(fig)

def main():
 style(); model,_=neural_bloch_inputs(); loaded={key:np.load(path) for key,path in FILES.items()}
 selected=selected_sectors()
 groups={filling:[(f'outer{m}',rf'v4 outer $P_{m}$'+(r' (lowest $E_{\rm tail}$)' if m==selected[filling] else '')) for m in range(3)] for filling in ('1/3','2/3')}
 rows=[(f,b,m) for f,g in groups.items() for b,m in g]; smax=max(float(loaded[(f,b)]['charge_structure_factor_full'].max()) for f,b,_ in rows)
 rhoall=np.concatenate([loaded[(f,b)]['charge_density_over_mean'].ravel() for f,b,_ in rows]); rmin,rmax=np.percentile(rhoall,[1,99]); FIG.mkdir(parents=True,exist_ok=True)
 for filling,methods in groups.items():
  fig,axs=plt.subplots(len(methods),3,figsize=(7.3,2.3*len(methods)+.7),gridspec_kw={'width_ratios':[1,1,1.22]}); fig.subplots_adjust(left=.125,right=.94,bottom=.09,top=.94,wspace=.36,hspace=.22); arts=[None,None,None]
  for row,(branch,method) in enumerate(methods):
   z=loaded[(filling,branch)]; kp=np.asarray([model.wrap_to_hexagon(k) for k in z['momentum_k_points']]); ax=axs[row,0]; hexagon(ax,model)
   arts[0]=ax.scatter(kp[:,0],kp[:,1],c=z['momentum_occupation_band1'],s=210,cmap='viridis',vmin=0,vmax=1,edgecolors='white',linewidths=.65,zorder=3,clip_on=False)
   ax=axs[row,1]; hexagon(ax,model); qp=np.asarray([model.wrap_to_hexagon(q) for q in z['structure_q_vectors']])
   arts[1]=ax.scatter(qp[:,0],qp[:,1],c=z['charge_structure_factor_full'],s=210,cmap='magma',vmin=0,vmax=smax,edgecolors='white',linewidths=.65,zorder=3,clip_on=False)
   ax=axs[row,2]; x=np.asarray(z['density_x_fraction']); y=np.asarray(z['density_y_fraction']); density=np.tile(np.asarray(z['charge_density_over_mean']),(2,2)); tx=np.concatenate([x,x+1]); ty=np.concatenate([y,y+1])
   fx,fy=np.meshgrid(tx,ty,indexing='ij'); cx=fx+.5*fy; cy=np.sqrt(3)*fy/2; arts[2]=ax.pcolormesh(cx,cy,density,shading='nearest',cmap='magma',vmin=rmin,vmax=rmax,rasterized=True,clip_on=False)
   ax.plot([0,2,3,1,0],[0,0,np.sqrt(3),np.sqrt(3),0],color='#34383B',lw=.75,clip_on=False); ax.set_aspect('equal'); ax.set_xlim(-.14,3.14); ax.set_ylim(-.12,np.sqrt(3)+.12)
   ax.set_xticks([0,1.5,3]); ax.set_yticks([0,np.sqrt(3)]); ax.set_yticklabels(['0',r'$\sqrt{3}$']); ax.set_xlabel(r'$x/a_M$'); ax.set_ylabel(r'$y/a_M$'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
   axs[row,0].text(-.37,.5,method,transform=axs[row,0].transAxes,rotation=90,ha='center',va='center',fontsize=8,fontweight='bold')
  for j,title in enumerate((r'$n_1(\mathbf{k})$',r'$S_{\rm full}(\mathbf{q})$',r'$\rho(\mathbf{r})/\bar\rho$')): axs[0,j].set_title(title,pad=5,fontweight='bold')
  fig.suptitle(rf'$\nu={filling}$',y=.985,fontsize=11,fontweight='bold')
  for label,ax in zip('abcdefghijkl',axs.flat): ax.text(-.11,1.03,label,transform=ax.transAxes,ha='left',va='bottom',fontsize=9,fontweight='bold')
  for j,(art,ticks) in enumerate(((arts[0],[0,.5,1]),(arts[1],[0,smax/2,smax]),(arts[2],[round(rmin,1),1,round(rmax,1)]))):
   pos=axs[-1,j].get_position(); cax=fig.add_axes([pos.x0,.022,pos.width,.009]); cb=fig.colorbar(art,cax=cax,orientation='horizontal'); cb.set_ticks(ticks); cb.ax.tick_params(length=2,pad=1); cb.outline.set_linewidth(.55)
  suffix='nu1of3' if filling=='1/3' else 'nu2of3'; fig.savefig(FIG/f'fig7_neural_bloch_observables_{suffix}.pdf',bbox_inches='tight',pad_inches=.08); fig.savefig(FIG/f'fig7_neural_bloch_observables_{suffix}.png',bbox_inches='tight',pad_inches=.08); plt.close(fig)

if __name__=='__main__':
 make_combined()
 main()
