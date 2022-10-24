from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import colors, ticker, cm
from matplotlib.colors import LogNorm
import matplotlib as mpl
import os


def D2_map(x,y,Z,level_min,level_max,log,title, xlabel,ylabel,filename,dpi=900): # function for 2 dimension plots
	"""
	Creates a 2D_map with nice settings
	x: 1D float array
		index
	y: 1D float array 
		columns 
	Z: 2D float array
		values
	level_min: float
		minimum value in the colorbar
	level_max:float 
		maximum value in the color bar
	log: string
		set "log" for a log scale colorbar
	xlabel: string
	ylabel: string
	filename: string
	dpi: int
		definition of the image to be saved
	"""

    
	from os import path
	#trying to make shift_graphs directory if it does not already exist:
	if not os.path.exists('figures'):
		os.mkdir('figures')
	filename=os.path.join("figures",filename)


		# Read input data from FITS file
	#mpl.rcParams['figure.dpi'] = 900
	plt.rcParams['axes.linewidth'] = 1
	
	#Y, X = np.meshgrid(x, y)
	Y, X = np.meshgrid(x, y)
	level_min=level_min
	level_max=level_max


	# Setup contour levels
	levels=np.arange(level_min,level_max,(level_max-level_min)/(100))
	

	fig_width_pt = 480.0  # Get this from LaTeX using \showthe\columnwidth
	inches_per_pt = 1.0/72.27				# Convert pt to inch
	golden_mean = (math.sqrt(5)-1.0)/2.0		 # Aesthetic ratio
	fig_width = fig_width_pt*inches_per_pt	# width in inches
	fig_height = fig_width*golden_mean		# height in inches
	fig_size =	[fig_width,fig_height]
	plt.rcParams['legend.fontsize'] = 8
	plt.rcParams['figure.figsize'] = fig_size
	#figure_style()
	
	
	fig, ax = plt.subplots()

	if log=="log":
		cs = ax.pcolormesh(Y, X, Z,vmin=level_min,vmax=level_max,shading='auto',  cmap='magma',	norm=LogNorm(),zorder=-20)
	else:
		cs = ax.pcolormesh(Y, X, Z,vmin=level_min,vmax=level_max,shading='auto',  cmap='magma',zorder=-20)
	ax.set_rasterization_zorder(-10)
	
	
	
	plt.xlim((x.min(), x.max()))
	plt.ylim((y.min(), y.max())) 
	plt.xlabel(xlabel, {'color': 'k', 'fontsize': 14})
	plt.ylabel(ylabel, {'color': 'k', 'fontsize': 14})
	plt.xticks( color='k', size=12)
	plt.yticks( color='k', size=12)
	
	v=np.linspace(level_min,level_max,num=2)
	norm = colors.Normalize(vmin=v.min(), vmax=v.max())
	axins1 = inset_axes(ax,
				   width="30%",	 # width = 5% of parent_bbox width
				   height="5%",	 # height : 50%
				   loc='lower left',
				   bbox_to_anchor=(0.7, 1.05, 1, 1),
				   bbox_transform=ax.transAxes,
				   borderpad=0,)
	cbar = fig.colorbar(cs,ticks= v,orientation="horizontal",cax=axins1)
	cbar.ax.tick_params(axis='x', labelsize=12, length=4) 
	cbar.ax.text(0,1.05,title,rotation=0,fontsize=14,fontweight=2,transform=ax.transAxes)
	axins1.xaxis.set_ticks_position("top")
	cbar.ax.set_xticklabels(['{:.4f}'.format(x) for x in v])
  



	# Setting number of ticks in x and y
	plt.locator_params(axis='y', nbins=5)
	plt.locator_params(axis='x', nbins=4)

	plt.savefig(filename+'.png',dpi=dpi, bbox_inches = "tight")


	plt.plot()
	