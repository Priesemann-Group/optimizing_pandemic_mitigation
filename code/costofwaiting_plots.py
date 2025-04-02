# Description: This script generates plots for the cost of waiting for mitigation in a disease model.
# Data generation is done in a separate script. (costofwaiting.py)

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp


# parameters
whichcost = 'div' # which mitigation cost function to use, options: 'div', 'divlog', 'divsqrt'
slope = 5 # slope of the infection cost function
a = 10 # disease severity 
gamma = 0.1 # recovery rate
years = 1 # number of years to simulate
T = years*365 # total time
dt = T/(T-1) # time step
deltas = np.arange(0,201,1) # delta range we want to look at for systematic plots
extra_deltas = (0,50,100) # example values for delta (need 4 values)
Mmax = 1 # x axis limit for mitigation plots

# path for saving of plot
path0 = 'data/costofwaiting_%s'%whichcost
date = 'costofwaiting_timedep_%gy_a%g/%s' %(years,a,whichcost) 


# mitigation cost functions
def CM_div(m):
    if whichcost  == "div":
        return	m/(1-m) # divergent cost 
    if whichcost == "divlog":
        return -jnp.log(1-m)
    if whichcost == "divsqrt":
        return (2/jnp.sqrt(1-m)-2)

# infection cost function
def CI_cost(a,I):
	h = 0.5*(5+jnp.sqrt(21)) 
	k = 0.001288575763893496
	return a * (1/h*I+h*(I-0.001)*(1-1/(1+jnp.exp(1000*(I-0.001))))+k)
     		
# adjust lightness of color
# adapted from https://gist.github.com/ihincks/6a420b599f43fcd7dbd79d56798c4e5a
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


# plotting function
def plot_costofwaiting(CMname,a,gamma, date,T,dt,deltas,extra_deltas,Mmax,a_s = [50,20,10,1]):
	
	# arrays for plotting
	Itot_timedep_plot = [] # total infections
	M_timedep_plot = [] # mitigation
	CI_timedep_plot = [] # infection cost
	CM_timedep_plot = [] # mitigation cost
	costtot_timedep_plot = [] # total cost

	# set font sizes
	plt.rc('font', size=10)
	plt.rc('axes', titlesize=10)
	plt.rc('axes', labelsize=8)
	plt.rc('legend', fontsize=8)
	plt.rc('xtick', labelsize=8)
	plt.rc('ytick', labelsize=8)
	plt.rc('figure', titlesize=10)

	# prepare figure
	(j1,j2,j3) = extra_deltas # use 4 values for delta for example plots
	fig, axs = plt.subplots(7,1, figsize =(6/2.54,16/2.54), height_ratios=[1,1,1,1,0.2,2.4,2]) # size in cm
	plt.subplots_adjust(wspace=4,hspace=0.4) # space between subplots
	
	# subplots
	ax0 = axs[5] 
	ax1 = axs[6]
	ax2 = axs[0]
	ax3 = axs[1]
	ax4 = axs[2]
	ax5 = axs[3]
	axs[4].set_visible(False) # remove empty subplot

	
	# twin axes for each subplot for prevalence and mitigation
	axC=ax2.twinx()
	axC.spines[['top']].set_visible(False)
	axD=ax3.twinx()
	axD.spines[['top']].set_visible(False)
	axE=ax4.twinx()
	# add one axis label for all example plots together
	axE.set_ylabel('prevalence (%)                    ', color = 'indianred', rotation = -90, labelpad = 10)
	ax4.set_ylabel('                      mitigation', color = 'cornflowerblue')
	axE.spines[['top']].set_visible(False)
	axF=ax5.twinx()
	axF.spines[['top']].set_visible(False)

	# load data for high delta so that we can observe infection peak before mitigation
	with open('%s/%s/CIslope%s_cost%s_delta%g_data.npy' %(path0,date,slope,CMname,deltas[-1]), 'rb') as g:
		time = np.load(g)
		S = np.load(g)
		I = np.load(g)
	peakindex = np.argmax(I) # find peak index
	peaktime = time[peakindex]
	print("peaktime",peaktime)
	peakbeginning = np.argmax(I>0.02)
	peakend = peakindex + np.argmin((I[peakindex::])>0.02)
	# mark infection wave in all plots
	ax0.axvspan(peakbeginning, peakend, color='whitesmoke', zorder = 0)
	ax0.axvline(x = peakindex, color = 'lightgrey', zorder = 0)
	ax1.axvspan(peakbeginning, peakend, color='whitesmoke', zorder = 0)
	ax1.axvline(x = peakindex, color = 'lightgrey', zorder = 0)
	ax2.axvspan(peakbeginning, peakend, color='whitesmoke', zorder = 0)
	ax2.axvline(x = peakindex, color = 'lightgrey', zorder = 0)
	ax3.axvspan(peakbeginning, peakend, color='whitesmoke', zorder = 0)
	ax3.axvline(x = peakindex, color = 'lightgrey', zorder = 0)
	ax4.axvspan(peakbeginning, peakend, color='whitesmoke', zorder = 0)
	ax4.axvline(x = peakindex, color = 'lightgrey', zorder = 0)
	ax5.axvspan(peakbeginning, peakend, color='whitesmoke', zorder = 0)
	ax5.axvline(x = peakindex, color = 'lightgrey', zorder = 0)
	
	
	for delta in deltas: # loop over different delays (needed for plotting)
		# load data
		with open('%s/%s/CIslope%s_cost%s_delta%g_data.npy' %(path0,date,slope,CMname,delta), 'rb') as g:
			time = np.load(g)
			S = np.load(g)
			I = np.load(g)
			M = np.load(g)
		
		# calculate cost
		CM = CM_div(M)
		CI = CI_cost(a,I)
		sumi = np.sum(I)*dt*gamma
		CM = sum(CM)*dt
		CI = sum(CI)*dt
		costtot = CI + CM	
		print(delta, sumi)
		
		# save data for plotting
		CI_timedep_plot.append(CI)
		CM_timedep_plot.append(CM)
		costtot_timedep_plot.append(costtot)
		M_timedep_plot.append(np.mean(M[int(delta)::]))
		Itot_timedep_plot.append(sumi)
		linestyle = '-'
		
		
		# panels b-d
		# example plots (prevalence and mitigation vs time) ---------------------------------------------------------------------------------------------------------------------
		until = 300 # x axis limit
		# example plots for certain deltas
		if delta == j1:
			# plot prevalence and mitigation
			axD.plot(time[:until],I[:until], label = 'I', color = adjust_lightness('indianred', amount=1), linestyle = linestyle)
			ax3.plot(time[:until],M[:until], label = 'M', color = adjust_lightness('cornflowerblue', amount=1), linestyle = linestyle)
			# axis labels etc.
			ax3.annotate(r'$\delta$ = %.0f' %(delta), xy=(0.05,0.7),xycoords='axes fraction',fontsize=8)
			ax3.set_ylim(-0.02,1)
			ax3.set_yticks([0,Mmax])
			axD.set_ylim(-0.02,0.3)
			axD.set_yticks([0,0.3])
			axD.set_yticklabels([0,30])
			ax3.set_xlim(0,until)
			axD.set_xlim(0,until)
			ax3.get_xaxis().set_ticks([0,100,200,until])
			ax3.tick_params(labelbottom=False) 
			sns.despine(ax=ax3, top=True, right=False, left=False, bottom=False, offset=2, trim=True)
			ax3.spines[['right', 'top']].set_visible(False)
			sns.despine(ax=axD, top=True, right=False, left=False, bottom=False, offset=2, trim=True)
			axD.spines[['left']].set_visible(False)
			for label in axD.get_yticklabels():
				label.set_color('indianred')
			for label in ax3.get_yticklabels():
				label.set_color('cornflowerblue')
		if delta == j2:
			# plot prevalence and mitigation
			axE.plot(time[:until],I[:until], label = 'I', color = adjust_lightness('indianred', amount=1), linestyle = linestyle)
			ax4.plot(time[:until],M[:until], label = 'M', color = adjust_lightness('cornflowerblue', amount=1), linestyle = linestyle)
			# axis labels etc.
			ax4.annotate(r'$\delta$ = %.0f' %(delta), xy=(0.05,0.7),xycoords='axes fraction',fontsize=8)
			ax4.set_ylim(-0.02,1)
			ax4.set_yticks([0,Mmax])
			axE.set_ylim(-0.02,0.3)
			axE.set_yticks([0,0.3])
			axE.set_yticklabels([0,30])
			ax4.set_xlim(0,until)
			axE.set_xlim(0,until)
			ax4.get_xaxis().set_ticks([0,100,200,until])
			ax4.tick_params(labelbottom=False) 
			sns.despine(ax=ax4, top=True, right=False, left=False, bottom=False, offset=2, trim=True)
			ax4.spines[['right', 'top']].set_visible(False)
			sns.despine(ax=axE, top=True, right=False, left=False, bottom=False, offset=2, trim=True)
			axE.spines[['left']].set_visible(False)
			for label in axE.get_yticklabels():
				label.set_color('indianred')
			for label in ax4.get_yticklabels():
				label.set_color('cornflowerblue')
		if delta == j3:
			# plot prevalence and mitigation
			axF.plot(time[:until],I[:until], label = 'I', color = adjust_lightness('indianred', amount=1), linestyle = linestyle)
			ax5.plot(time[:until],M[:until], label = 'M', color = adjust_lightness('cornflowerblue', amount=1), linestyle = linestyle)
			# axis labels etc.
			ax5.annotate(r'$\delta$ = %.0f' %(delta), xy=(0.05,0.7),xycoords='axes fraction',fontsize=8)
			ax5.set_ylim(-0.02,1)
			axF.set_ylim(-0.02,0.3)
			ax5.set_yticks([0,Mmax])
			axF.set_yticks([0,0.3])
			axF.set_yticklabels([0,30])
			ax5.set_xlabel('time (days)')
			ax5.set_xlim(0,until)
			axF.set_xlim(0,until)
			ax5.get_xaxis().set_ticks([0,100,200,until])
			labels = [item.get_text() for item in ax5.get_xticklabels()]
			labels = ['0','100','200','300']
			ax5.set_xticklabels(labels)
			sns.despine(ax=ax5, top=True, right=False, left=False, bottom=False, offset=2, trim=True)
			ax5.spines[['right', 'top']].set_visible(False)
			sns.despine(ax=axF, top=True, right=False, left=False, bottom=False, offset=2, trim=True)
			axF.spines[['left']].set_visible(False)
			for label in axF.get_yticklabels():
				label.set_color('indianred')
			for label in ax5.get_yticklabels():
				label.set_color('cornflowerblue')
	
		# panel a
		# plot infections with no mitigation ----------------------------------------------------------------------------------
		with open('%s/%s/CIslope%s_cost%s_nomit_data.npy' %(path0,date,slope,CMname), 'rb') as g:
			time = np.load(g)
			S = np.load(g)
			I = np.load(g)
			M = np.zeros(len(I))
		
		axC.plot(time[:until],I[:until], label = 'I', color = adjust_lightness('indianred', amount=1), linestyle = linestyle)
		ax2.plot(time[:until],M[:until], label = 'M', color = adjust_lightness('cornflowerblue', amount=1), linestyle = linestyle)
		ax2.set_ylim(-0.02,1)
		ax2.set_yticks([0,Mmax])
		axC.set_ylim(-0.02,0.3)
		axC.set_yticks([0,0.3])
		axC.set_yticklabels([0,30])
		ax2.annotate('no mit.', xy=(0.02,0.7),xycoords='axes fraction',fontsize=8)
		ax2.set_xlim(0,until)
		axC.set_xlim(0,until)
		ax2.get_xaxis().set_ticks([0,100,200,until])
		ax2.tick_params(labelbottom=False) 
		sns.despine(ax=ax2, top=True, right=False, left=False, bottom=False, offset=2, trim=True)
		ax2.spines[['right', 'top']].set_visible(False)
		sns.despine(ax=axC, top=True, right=False, left=False, bottom=False, offset=2, trim=True)
		axC.spines[['left']].set_visible(False)
		ax2.text(0.32,0.6,'infection peak\nwithout mit.', color = 'darkgrey',fontsize = 7, transform=ax2.transAxes)
		for label in axC.get_yticklabels():
			label.set_color('indianred')
		for label in ax2.get_yticklabels():
			label.set_color('cornflowerblue')
		
	# panel e	
	# cost vs delta ---------------------------------------------------------------------------------------------------------------------
	# load data for no mitigation
	with open('%s/%s/CIslope%s_cost%s_nomit_data.npy' %(path0,date,slope,CMname), 'rb') as g:
		time = np.load(g)
		S = np.load(g)
		I = np.load(g)
		M = np.zeros(len(I))
	# calculate cost
	CM = CM_div(M)
	CI = CI_cost(a,I)
	sumi = np.sum(I)*dt*gamma
	CM = sum(CM)*dt
	CI = sum(CI)*dt
	costtot = CI + CM	
	
	ax0.axhline(y = costtot/len(I), color = 'k', linestyle = '--') # mark cost for no mitigation
	# costs for different deltas
	ax0.plot(deltas,np.array(costtot_timedep_plot)/len(I), color = 'k') # total cost
	ax0.plot(deltas,np.array(CM_timedep_plot)/len(I), color = adjust_lightness('cornflowerblue', amount=1)) # mitigation cost
	ax0.plot(deltas,np.array(CI_timedep_plot)/len(I), color = adjust_lightness('indianred', amount=1)) # infection cost
	# label lines
	ax0.text(1.01, 0.02, 'mitigation \ncost', color = 'cornflowerblue',fontsize = 8, transform=ax0.transAxes)
	ax0.text(1.01, 0.59, 'total \ncost', color = 'k',fontsize = 8, transform=ax0.transAxes)
	ax0.text(1.01, 0.34, 'infection \ncost', color = 'indianred',fontsize = 8, transform=ax0.transAxes)
	ax0.text(0.07, 0.9, 'without mit.', color = 'k',fontsize = 8, transform=ax0.transAxes)
	# add axis labels etc.
	ax0.set_ylabel('mean daily\ncost (a.u.)   ') # y axis label
	ax0.set_xlim(0,200) # x axis limit
	ax0.get_xaxis().set_ticks([0,50,100,150,200]) # x axis ticks
	ax0.tick_params(labelbottom=False) # remove x axis ticks (same as apenl below, only show once)
	ax0.set_ylim(0,2.5) # y axis limit
	ax0.get_yaxis().set_ticks([0,1,2]) # y axis ticks
	ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '%g' % x)) # remove decimal places
	ax0.spines[['right', 'top']].set_visible(False) # remove spines
	
	# panel f
	# comparison different disease severities ---------------------------------------------------------------------------------------------------------------------
	ax= ax1
	colors = ["k", 'dimgrey', 'darkgrey','lightgrey'] # colors for different a
	for i,a_ in enumerate(a_s): # loop over different a
		Ctots = [] # array for cost per day
		for delta in deltas: # loop over different deltas
			# load data
			date = 'costofwaiting_timedep_%gy_a%g/%s' %(years,a_,whichcost) 
			with open('%s/%s/CIslope%s_cost%s_delta%g_data.npy' %(path0,date,slope,CMname,delta), 'rb') as g:
				time = np.load(g)
				S = np.load(g)
				I = np.load(g)
				M = np.load(g)
			# calculate cost
			CM = CM_div(M)
			CI = CI_cost(a_,I)
			CM = sum(CM)*dt
			CI = sum(CI)*dt
			costtot = CI + CM	
			costtot = costtot / len(I) # cost per day
			Ctots.append(costtot) # append cost to array
		# plot cost for this a
		ax.plot(deltas,Ctots, label = r'$a=$%g'%(a_), color = colors[i])
	# add axis labels etc.
	ax.set_xlabel(r'mitigation delay $\delta$ (days)') # x axis label
	ax.set_ylabel('mean daily\ntotal cost (a.u.)') # y axis label
	ax.legend(loc='upper left', bbox_to_anchor=(0.98, 1), fontsize=8, frameon=False, handlelength = 0.8) # legend
	ax.set_xlim(0,200) # x axis limit
	ax.get_xaxis().set_ticks([0,50,100,150,200]) # x axis ticks
	ax.tick_params(labelbottom=True) # show x axis ticks
	ax.set_ylim(0,8) # y axis limit
	ax.get_yaxis().set_ticks([0,4,8]) # y axis ticks
	ax.spines[['right','top']].set_visible(False) # remove spines
	# add break in y axis because it doesn't start at 0
	#xpos = 0 # x position of the "break"
	#ypos = 0.17  # y position of the "break"
	#ax1.scatter(xpos+1,ypos, color='white', marker='s', s=80, clip_on=False, zorder=100)
	#ax1.text(xpos-3.1, ypos, r'...', rotation = 90, fontsize=9, zorder=101, horizontalalignment='center', verticalalignment='center')

	
	
	# label panels -------------------------------------------------------------------------------------------------
	xplace = -0.25
	yplace = 1.1
	ax0.text(xplace,yplace-0.08, 'e', transform=ax0.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
	ax1.text(xplace,yplace-0.08, 'f', transform=ax1.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
	ax2.text(xplace,yplace, 'a', transform=ax2.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
	ax3.text(xplace,yplace, 'b', transform=ax3.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
	ax4.text(xplace,yplace, 'c', transform=ax4.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
	ax5.text(xplace,yplace, 'd', transform=ax5.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
	
	# save figure
	fig.savefig("%s/costofwaiting_comparison_%s_a%g.pdf" %(path0,CMname,a), bbox_inches="tight")
	



# plot ------------------------------------------------------------------------------------------------------------
plot_costofwaiting(whichcost,a,gamma, date,T,dt,deltas,extra_deltas,Mmax)
