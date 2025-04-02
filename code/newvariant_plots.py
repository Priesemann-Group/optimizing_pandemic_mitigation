# Description: This script generates plots for the new variant scenario.
# Data is generated from the newvariant_optimal.py and newvariant_late_adjustment.py scripts.

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.legend_handler import HandlerBase

# parameters --------------------------------------------------------------------------------------------------------------------
whichcost = 'div' # which cost function to use, options are 'div', 'divlog', 'divsqrt'
beta_0_1 = 0.2 # transmission rate for variant 1
beta_0_2 = 0.3 # transmission rate for variant 2
gamma = 0.1 # recovery rate
yearstot = 9 # years for simulation
a = 10 # disease severity


# where to get data and save plot
path00 = './data/newvariant_multcomp_%s/'%whichcost
path00_unknown = './data/newvariant_multcomp_%s/'%whichcost
path00_known = './data/newvariant_multcomp_%s/'%whichcost

# helper functions and plot settings --------------------------------------------------------------------
class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                           linestyle=orig_handle[1], color='k')
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], 
                           linestyle=orig_handle[0], color = 'indianred')
        return [l1, l2]

# font settings
plt.rc('font', size=8)
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('figure', titlesize=10)


# infection cost function
def CI_cost(a,I):
	h = 0.5*(5+np.sqrt(21)) # slope 5
	k = 0.001288575763893496
	return a*(1/h*I+h*(I-0.001)*(1-1/(1+np.exp(1000*(I-0.001))))+k)

# mitigation cost function
def CM_div(m):
    if whichcost  == "div":
        return	m/(1-m) # divergent cost 
    if whichcost == "divlog":
        return -np.log(1-m)
    if whichcost == "divsqrt":
        return (2/np.sqrt(1-m)-2)

# plot function -----------------------------------------------------------------------------------------------------------------
def plot_all(a, beta_0_1, beta_0_2):
     
    # preparation -------------------------------------------------------------------------------------`
    years = 8 # years we want to plot
    datafrom = 0 # from when we want to plot the data
    datato = years*365 # until when we want to plot the data
    CM_name = whichcost # name of the cost function
    # paths where data is saved
    date_unknown = 'newvariant_unknown_%gyr_a%g' %(yearstot,a)
    path0_unknown = path00_unknown + date_unknown + '/'
    date_known = 'newvariant_known_%gyr_a%g' %(yearstot,a)
    path0_known = path00_known + date_known + '/' 
    path_known = path0_known + CM_name + '/'
    path_unknown = path0_unknown + CM_name + '/'

    

    # plot settings
    xsize = 7/2.45
    ysize = 17/2.45
    fig = plt.figure(figsize=(xsize, ysize))
    
    # create plot layout
    gs = gridspec.GridSpec(5,1, height_ratios=[1,0.2,3,0.07,1.1], hspace = 0.4, figure = fig)
    gs1 = gs[0].subgridspec(1,1)
    gs2 = gs[2].subgridspec(1,1)
    gs3 = gs[4].subgridspec(1,1)
    gsexamples = gs2[0].subgridspec(8,2, width_ratios=[1.2,1], height_ratios=[1,1,0.2,1,1,0.2,1,1], hspace=0.5, wspace=0.05)
    gsopt = gs1[0].subgridspec(2,1, hspace = 0.5)
    gscost = gs3[0].subgridspec(1,2, width_ratios=[1,1], wspace=1.2)

    # add plots to layout
    ax_optMI = fig.add_subplot(gsopt[0]) # optimal 
    ax_optR = fig.add_subplot(gsopt[1]) 

    ax_longIM_1 = fig.add_subplot(gsexamples[0,1]) # unknown examples long
    ax_longR_1 = fig.add_subplot(gsexamples[1,1])
    ax_longIM_2 = fig.add_subplot(gsexamples[3,1])
    ax_longR_2 = fig.add_subplot(gsexamples[4,1])
    ax_longIM_3 = fig.add_subplot(gsexamples[6,1])
    ax_longR_3 = fig.add_subplot(gsexamples[7,1])

    ax_shortIM_1 = fig.add_subplot(gsexamples[0,0]) # unknown examples short
    ax_shortR_1 = fig.add_subplot(gsexamples[1,0])
    ax_shortIM_2 = fig.add_subplot(gsexamples[3,0])
    ax_shortR_2 = fig.add_subplot(gsexamples[4,0])
    ax_shortIM_3 = fig.add_subplot(gsexamples[6,0])
    ax_shortR_3 = fig.add_subplot(gsexamples[7,0])

    ax_totalcost = fig.add_subplot(gscost[0]) # costs
    ax_totalcostdifferenta = fig.add_subplot(gscost[1]) # costs for different a

    
   
    # optimal-----------------------------------------------------------------------------------
    # infections, mitigation, effective reproduction number
    # load data for optimal scenario
    with open('%sa%g_beta1_%g_beta2_%g_multcomp_data.npy' %(path_known,a,beta_0_1,beta_0_2), 'rb') as g: # CHANGE
        t = np.load(g)
        S = np.load(g)
        I1 = np.load(g)
        I2 = np.load(g)
        M = np.load(g)

    # calculate cost
    I = np.add(I1,I2)
    CM = CM_div(M)
    dt = 1
    CI = CI_cost(a,I)
    CM = sum(CM)*dt
    CI = sum(CI)*dt
    costtot = CI + CM	
    optcost = (costtot)/len(I) # mean daily cost
    print('optimal cost',optcost)

    # data preparation
    t = t[datafrom:datato]
    I1 = I1[datafrom:datato]
    I2 = I2[datafrom:datato]
    M = M[datafrom:datato]

    # which axes to use
    ax = ax_optMI
    axR = ax_optR
    ax2 = ax.twinx() # twin ax for infections
    ax.text(-0.25,1.9,'a',fontsize=8, weight = 'bold',transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', boxstyle='circle,pad=0.1')) # panel label
    ax.tick_params(labelbottom=False) # no x ticks
    ax.set_ylim(0,1) # y limits
    ax.set_yticks([0,1]) # y ticks
    ax2.set_ylim(-0.00001,0.003) # y limits for infections
    yticks = [0,0.003] # y ticks for infections
    ax2.set_yticks(yticks) 
    ax2.set_yticklabels([0,0.3]) # y tick labels for infections (*100)
    ax2.set_ylabel('preval.\n'+r'$I$ (%)', color = 'indianred', rotation = -90, labelpad=20) # y label for infections
    ax.set_ylabel('miti-\ngation\n'+r'$M$', color = 'cornflowerblue') # y label for mitigation
    ax2.plot(t, I1, color = 'indianred')   # plot infections for variant 1
    ax2.plot(t, I2, color = 'indianred', linestyle='--') # plot infections for variant 2
    ax2.legend([("-","-"), ("--","--")], [r'var. 1, $R_0 = 2$', r"var. 2, $R_0=3$"], handler_map={tuple: AnyObjectHandler()}, loc='lower left', bbox_to_anchor=(0.1, 1), ncol=2, fontsize=7, frameon=False) #legend
    ax.plot(t, M, label='M', color = 'cornflowerblue') # plot mitigation
    ax.set_title('optimal scenario (new variant known from '+r'$t=0$)', y = 1.7) # title
    axR.set_xlim(datafrom,datato) # x limits
    axR.set_xticks([int(datafrom),int((datafrom+datato)/4),int((datafrom+datato)/2),int(3*(datafrom+datato)/4),int(datato)]) # x ticks
    ax.set_xlim(datafrom,datato) # x limits
    ax.set_xticks([int(datafrom),int((datafrom+datato)/4),int((datafrom+datato)/2),int(3*(datafrom+datato)/4),int(datato)]) # x ticks
    axR.set_xticklabels([int(datafrom/365),int((datafrom+datato)/4/365),int((datafrom+datato)/2/365),int(3*(datafrom+datato)/4/365),int(datato/365)])
    ax.set_xticklabels([int(datafrom/365),int((datafrom+datato)/4/365),int((datafrom+datato)/2/365),int(3*(datafrom+datato)/4/365),int(datato/365)])
    sns.despine(ax=ax, top=True, right=False, left=False, bottom=False, offset=0.2, trim=True) # remove top and right spines
    sns.despine(ax=ax2, top=True, right=False, left=False, bottom=False, offset=0.2, trim=True)
    for label in ax2.get_yticklabels(): # change color of y tick labels
        label.set_color('indianred')
    for label in ax.get_yticklabels():
        label.set_color('cornflowerblue')
    R1 = 1/0.1*beta_0_1*(1-M) # effective reproduction number for variant 1
    R2 = 1/0.1*beta_0_2*(1-M) # effective reproduction number for variant 2
    axR.plot(t, R1, color = 'k') # plot effective reproduction number for variant 1
    axR.plot(t, R2, color = 'k', linestyle='--') # plot effective reproduction number for variant 2
    axR.set_ylabel('effective     \nreprod.    \nnumber    \n'+r'$R_{\text{eff}}$    ', color = 'k') # y label for effective reproduction number
    axR.set_ylim(0,2) # y limits
    axR.set_yticks([0,1,2]) # y ticks
    axR.axhline(y=1, color='lightgrey', linestyle='-', zorder = 0) # horizontal line at r=1
    axR.tick_params(labelbottom=True) # x ticks
    sns.despine(ax=axR, top=True, right=True, left=False, bottom=False, offset=0, trim=True) # remove top and right spines
    ax2.tick_params(labelbottom=True) # x ticks
    axR.set_xlabel('time $t$ (years)') # x label


    # late adjustment plots------------------------------------------------------------------------

    # preparation
    abc = ['b','c','d'] # panel labels
    # we need a short and a long axis to get the effect of the middle part being cut out
    ax2s = [] # list for twin axes
    ax2s_short = [] # list for twin axes
    count = 0 # counter for loop
    maxIs = [0.003,0.003,0.1] # y axis limits for infections

    # loop over different mitigation change times
    for (ax_long,axR_long,ax_short,axR_short),t_mitchange in zip([(ax_longIM_1,ax_longR_1,ax_shortIM_1,ax_shortR_1),(ax_longIM_2,ax_longR_2,ax_shortIM_2,ax_shortR_2),(ax_longIM_3,ax_longR_3,ax_shortIM_3,ax_shortR_3)],[100,200,300]):
        
        maxI = maxIs[count] # y axis limit for infections

        # load data for different mitigation change times
        with open('%stmitchange%g_beta1_%g_beta2_%g_multcomp_data.npy' %(path_unknown,t_mitchange,beta_0_1,beta_0_2), 'rb') as g: # CHANGE
            t_complete = np.load(g)
            S = np.load(g)
            I_1complete = np.load(g)
            I_2complete = np.load(g)
            M_complete = np.load(g)

        # calculate cost
        I_cost = np.add(I_1complete,I_2complete) # I = I1 + I2
        CM = CM_div(M_complete)
        dt = 1
        CI = CI_cost(a,I_cost)
        CM = sum(CM)*dt
        CI = sum(CI)*dt
        costtot = (CI + CM) / len(I_cost) # mean daily cost
        
        ax_totalcost.scatter(t_mitchange,costtot, color='k', s=8) # make a scatter in the cost plot to mark this example value ther

        # plots long------------------------------------------------------------------------
        # preparation
        t_8years = t_complete[int(5.7*365):8*365]
        I1_8years = I_1complete[int(5.7*365):8*365]
        I2_8years = I_2complete[int(5.7*365):8*365]
        M_8years = M_complete[int(5.7*365):8*365] 

        # print information
        print("t_mitchange",t_mitchange,"-------------------")
        print("I1 after 8 years",I1_8years[-1])
        print("I2 after 8 years",I2_8years[-1])
        print("M after 8 years",M_8years[-1])

        # plot data
        if count == 0: # infromation text
            ax_short.text(0.28,1.5,'mitigation is adjusted only at '+r'$t_{\text{adj}}$',fontsize=8,transform=ax_short.transAxes)
            ax_short.text(0.15,1.1,r'$t_{\text{adj}}$',fontsize=8,color = 'grey',transform=ax_short.transAxes)
        ax2_long = ax_long.twinx() # twin axis for infections
        ax2s.append(ax2_long) # append twin axis to list
        ax_long.tick_params(labelbottom=False) # no x ticks
        ax_long.set_ylim(0,1) # y limits
        ax_long.set_yticks([]) # y ticks
        ax2_long.set_ylim(0,maxI) # y limits for infections
        yticks = [0,maxI] # y ticks for infections
        ax2_long.set_yticks(yticks) # y ticks for infections
        ax2_long.set_yticklabels(['0','%g'%(maxI*100)]) # y tick labels for infections (*100)
        ax2_long.set_ylabel(r'$I$ (%)', color = 'indianred', rotation = -90, labelpad=10) # y label for infections
        ax2_long.plot(t_8years, I1_8years, label='I1', color = 'indianred') # plot infections for variant 1
        ax2_long.plot(t_8years, I2_8years, label='I2', color = 'indianred', linestyle='--') # plot infections for variant 2
        ax_long.plot(t_8years, M_8years, label='M', color = 'cornflowerblue') # plot mitigation
        ax_long.set_xlim(int(5.7*365),8*365) # x limits
        ax_long.set_xticks([6*365,7*365,8*365]) # x ticks
        ax_long.set_xticklabels([6,7,8]) # x tick labels (years)
        ax_short.text(-0.45,1,'%s' %(abc[count]),fontsize=8, weight = 'bold',transform=ax_short.transAxes, bbox=dict(facecolor='white', edgecolor='white', boxstyle='circle,pad=0.1')) # panel label
        sns.despine(ax=ax_long, top=True, right=False, left=True, bottom=False, offset=0, trim=False) # remove top and right spines
        sns.despine(ax=ax2_long, top=True, right=False, left=True, bottom=False, offset=0, trim=False)
        R1_8years = 1/0.1*beta_0_1*(1-M_8years) # effective reproduction number for variant 1
        R2_8years = 1/0.1*beta_0_2*(1-M_8years) # effective reproduction number for variant 2
        axR_long.plot(t_8years, R1_8years, color = 'k') # plot effective reproduction number for variant 1
        axR_long.plot(t_8years, R2_8years, color = 'k', linestyle='--') # plot effective reproduction number for variant 2
        axR_long.set_ylim(0,2.1) # y limits
        axR_long.set_yticks([]) # y ticks
        axR_long.axhline(y=1, color='lightgrey', linestyle='-', zorder = 0) # horizontal line at R=1
        axR_long.set_xlim(int(5.7*365),8*365) # x limits
        axR_long.set_xticks([6*365,7*365,8*365]) # x ticks
        axR_long.set_xticklabels([6,7,8]) # x tick labels (years)
        axR_long.tick_params(labelbottom=False) # no x ticks
        for label in ax2_long.get_yticklabels(): # change color of y tick labels
            label.set_color('indianred')
        for label in ax_long.get_yticklabels():
            label.set_color('cornflowerblue')
        sns.despine(ax=axR_long, top=True, right=True, left=True, bottom=False, offset=0, trim=False) # remove top and right spines
        
    
        # plots short------------------------------------------------------------------------ 

        maxI = maxIs[count] # y axis limit for infections

        # load data for different mitigation change times
        with open('%stmitchange%g_beta1_%g_beta2_%g_multcomp_data.npy' %(path_unknown,t_mitchange,beta_0_1,beta_0_2), 'rb') as g: # CHANGE
            t = np.load(g)
            S = np.load(g)
            I1 = np.load(g)
            I2 = np.load(g)
            M = np.load(g)

        # preparation
        t_1yr = t_complete[:int(1.8*365)] 
        I1_1yr = I_1complete[:int(1.8*365)]
        I2_1yr = I_2complete[:int(1.8*365)]
        M_1yr = M_complete[:int(1.8*365)]

        ax2_short = ax_short.twinx() # twin axis for infections
        ax2s_short.append(ax2_short) # append twin axis to list
        ax_short.axvline(x=t_mitchange, color='grey', linestyle='-', zorder = 0) # line where mitigation is changed
        axR_short.axvline(x=t_mitchange, color='grey', linestyle='-', zorder = 0) # line where mitigation is changed
        ax_short.tick_params(labelbottom=False)  # no x ticks
        ax_short.set_ylim(0,1) # y limits
        ax_short.set_yticks([0,1]) # y ticks
        ax2_short.set_ylim(0,maxI) # y limits for infections
        ax2_short.set_yticks([]) # y ticks for infections
        ax_short.set_ylabel(r'$M$', color = 'cornflowerblue') # y label for mitigation
        ax2_short.plot(t_1yr, I1_1yr, label='I1', color = 'indianred') # plot infections for variant 1
        ax2_short.plot(t_1yr, I2_1yr, label='I2', color = 'indianred', linestyle='--') # plot infections for variant 2
        ax_short.plot(t_1yr, M_1yr, label='M', color = 'cornflowerblue') # plot mitigation
        ax_short.set_xlim(0,int(1.8*365)) # x limits
        ax_short.set_xticks([0,365/2,365,3*365/2]) # x ticks
        ax_short.set_xticklabels(['0','0.5','1','1.5']) # x tick labels (years)
        sns.despine(ax=ax_short, top=True, right=True, left=False, bottom=False, offset=0, trim=False) # remove top and right spines
        sns.despine(ax=ax2_short, top=True, right=True, left=False, bottom=False, offset=0, trim=False)
        R1_1yr = 1/0.1*beta_0_1*(1-M_1yr) # effective reproduction number for variant 1
        R2_1yr = 1/0.1*beta_0_2*(1-M_1yr) # effective reproduction number for variant 2
        axR_short.plot(t_1yr, R1_1yr, color = 'k') # plot effective reproduction number for variant 1
        axR_short.plot(t_1yr, R2_1yr, color = 'k', linestyle='--') # plot effective reproduction number for variant 2
        axR_short.set_ylabel(r'$R_{\text{eff}}$', color = 'k') # y label for effective reproduction number
        axR_short.set_ylim(0,2.1) # y limits
        axR_short.set_yticks([0,1,2]) # y ticks
        axR_short.axhline(y=1, color='lightgrey', linestyle='-', zorder = 0) # horizontal line at R=1
        axR_short.set_xlim(0,int(1.8*365)) # x limits
        axR_short.set_xticks([0,365/2,365,3*365/2]) # x ticks
        axR_short.set_xticklabels(['0','0.5','1','1.5']) # x tick labels (years)
        axR_short.tick_params(labelbottom=False) # no x ticks
        sns.despine(ax=axR_short, top=True, right=True, left=False, bottom=False, offset=0, trim=False)  # remove top and right spines
        for label in ax2_short.get_yticklabels(): # change color of y tick labels
            label.set_color('indianred')
        for label in ax_short.get_yticklabels():
            label.set_color('cornflowerblue')

        # part between short and long plot
        d = 0.6  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=7,linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_short.plot([1], [0], transform=ax_short.transAxes, **kwargs)
        ax_long.plot([0], [0], transform=ax_long.transAxes, **kwargs)
        axR_short.plot([1], [0], transform=axR_short.transAxes, **kwargs)
        axR_long.plot([0], [0], transform=axR_long.transAxes, **kwargs)

        count += 1 # increase counter



    # cost ---------------------------------------------------------------------------------------------------
    # mean daily total cost as a function of waiting time

    # preparation
    xmax =500 # until when we want to plot the cost
    ts_mitchange = np.arange(5,xmax+1,5) # time points for mitigation change
    costs = [] # list for costs
    ax = ax_totalcost # which axis to use

    # loop over different mitigation change times and calculate cost
    for t_mitchange in ts_mitchange:
        # load data for different mitigation change times
        with open('%stmitchange%g_beta1_%g_beta2_%g_multcomp_data.npy' %(path_unknown,t_mitchange,beta_0_1,beta_0_2), 'rb') as g: 
            t = np.load(g)
            S = np.load(g)
            I1 = np.load(g)
            I2 = np.load(g)
            M = np.load(g)

        I = np.add(I1,I2) # I = I1 + I2
        CM = CM_div(M)
        dt = 1
        CI = CI_cost(a,I)
        CM = sum(CM)*dt
        CI = sum(CI)*dt
        costtot = (CI + CM) / len(I) # mean daily cost
        costs.append(costtot)
        if t_mitchange == 30 or t_mitchange == 60: # print cost for t_mitchange = 30 and 60 (needed for paper)
            print('%g days I'%t_mitchange,np.sum(I)*dt*0.1)
            print('%g days cost'%t_mitchange,costtot)

    
    ax.axhline(y=optcost, color='k', linestyle=':', zorder = 10)
    ax.annotate('optimal (a)' %(costtot), xy=(0.4,0.2),xycoords='axes fraction',fontsize=7)
    ax.plot(ts_mitchange, costs, color = 'k', zorder = 10)
    ax.set_ylabel('total cost', color = 'k')    
    ax.set_xlabel(r'                                             mitigation adjustment time $t_\text{adj}$ (days)', color = 'k') # x label for cost     
    ax.set_ylim(1.8,2) # y limits  
    ax.set_yticks([1.9,2]) # y ticks                                    
    ax.set_xlim(0,xmax)
    ax.get_xaxis().set_ticks([0,xmax/2,xmax])
    ax.tick_params(labelbottom=True)
    ax.text(0.18,0.48,'b',horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)
    ax.text(0.39,0.55,'c',horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)
    ax.text(0.58,0.72,'d',horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, offset=0, trim=False)
    ax.text(-0.75,1,'e',fontsize=8, weight = 'bold',transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', boxstyle='circle,pad=0.1'))
    
    # break in y axis becaus it doesn't start at 0
    ax.text(-19, 1.83, "...", fontsize=10, color="black",ha="center", va="center", rotation=90,bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.1"))


    # cost for different a ------------------------------------------------------------------------------------

    ax2 = ax_totalcostdifferenta # which axis to use
    a_values = [50,20,10,1] # different a values
    costs_a = [[], [], [], []] # list for costs

    # loop over different a values
    for i,a_ in enumerate(a_values):
        # load data
        path_a = path00_unknown + 'newvariant_unknown_%gyr_a%g' %(yearstot,a_) + '/' + CM_name + '/'
        for t_mitchange in ts_mitchange:
            # load data for different mitigation change times
            with open('%stmitchange%g_beta1_%g_beta2_%g_multcomp_data.npy' %(path_a,t_mitchange,beta_0_1,beta_0_2), 'rb') as g: 
                t = np.load(g)
                S = np.load(g)
                I1 = np.load(g)
                I2 = np.load(g)
                M = np.load(g)

            I = np.add(I1,I2) # I = I1 + I2
            CM = CM_div(M)
            dt = 1
            CI = CI_cost(a_,I)
            CM = sum(CM)*dt
            CI = sum(CI)*dt
            costtot = (CI + CM) / len(I) # mean daily cost
            costs_a[i].append(costtot) # append cost to list

    colors = ["k", 'dimgrey', 'darkgrey','lightgrey'] # colors for different a
    for i,a_ in enumerate(a_values):
        ax2.plot(ts_mitchange, costs_a[i], color = colors[i], label = r'$a=%g$' %(a_), zorder = 10)
    ax2.set_ylabel('mean daily\ntotal cost (a.u.)', color = 'k') # y label for cost
    ax2.set_ylim(0,3) # y limits
    ax2.set_xlim(0,xmax) # x limits
    ax2.get_xaxis().set_ticks([0,xmax/2,xmax]) # x ticks
    ax2.tick_params(labelbottom=True) # x ticks
    sns.despine(ax=ax2, top=True, right=True, left=False, bottom=False, offset=0, trim=True) # remove top and right spines
    ax2.text(-0.75,1,'f',fontsize=8, weight = 'bold',transform=ax2.transAxes, bbox=dict(facecolor='white', edgecolor='white', boxstyle='circle,pad=0.1'))
    ax2.legend(loc='lower left', bbox_to_anchor=(0.1, 0.8), fontsize=7, frameon=False, handlelength=1, handletextpad=0.5, borderpad=0.5, ncol=2) # legend
    



    
    # aesthetics
    fig.align_ylabels(ax2s_short) # align y labels
    axR_short.tick_params(labelbottom=True) # x ticks
    axR_short.set_xlabel(r'time $t$ (years)', x=1) # x label for all plots
    fig.align_ylabels(ax2s) # align y labels
    axR_long.tick_params(labelbottom=True) # x ticks

    # save plot
    fig.savefig('%splot_newvariant_%s_%gyr_a%g_beta1_%g_beta2_%g_multcomp.pdf' %(path00,whichcost,yearstot,a,beta_0_1,beta_0_2), bbox_inches="tight", dpi = 600)


# call plot funciton ---------------------------------------------------------------------------------
plot_all(a, beta_0_1, beta_0_2)