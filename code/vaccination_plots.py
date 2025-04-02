# Description: This script generates plots for the vaccination model.
# Data generation is done in the script vaccination.py.

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches

# parameters for which to plot
whichcost = 'div' # which cost function to use, can be 'div', 'divlog', 'divsqrt'
beta_0_1 = 0.2 # basic reproduction number
vacc_eff = 1 # vaccination efficiency
gamma = 0.1 # recovery rate
yearstot = 3 # number of years for data
a = 10 # disease severity
nuv = 1/150 # waning rate for vaccination

# path where data is saved
path0 = './data/vacc_multcomp_%s_0.4max_%gyears/'%(whichcost,yearstot)

# set-up plotting ----------------------------------------------------------------------------------
# set font sizes
plt.rc('font', size=8)
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('figure', titlesize=10)


# infection cost
def CI_cost(a,I):
	h = 0.5*(5+np.sqrt(21)) # slope 5
	k = 0.001288575763893496
	return a * (1/h*I+h*(I-0.001)*(1-1/(1+np.exp(1000*(I-0.001))))+k)

# mitigation cost
def CM_div(m):
    if whichcost  == "div":
        return	m/(1-m) # divergent cost 
    if whichcost == "divlog":
        return -np.log(1-m)
    if whichcost == "divsqrt":
        return (2/np.sqrt(1-m)-2)

# plotting function
def plot_vacc(path0, a, beta_0_1,nuv):
     
    years = yearstot # number of years for data
    T = years*365 # number of days for data
    Tshow = 2*365 # number of days to show in plot (x axis range)
    labels = np.array([0,1,2]) # labels for x axis (years)
    datafrom = 0 # starting point x axis
    datato = Tshow # end point x axis
    dt = yearstot*365/(yearstot*365-1) # time step
    CM_name = whichcost # name of cost function
    # path
    date_known = 'vacc_%gyr_a%g' %(years,a)
    path0_known = path0 + date_known + '/' 
    path_known = path0_known + CM_name + '/'

    # plot set-up
    xsize = 15.5/2.54 # width in inches (inch = cm/2.54)
    ysize = 7/2.54
    fig = plt.figure(figsize=(xsize, ysize))

    # set up grid for subplots
    gs = gridspec.GridSpec(2,3, width_ratios=[1,1,1], height_ratios=[1,1],wspace = 0.8, hspace = 0.8, figure = fig)
    ax00 = fig.add_subplot(gs[0,0]) # vaccination rate
    ax01 = fig.add_subplot(gs[0,1]) # susceptible
    ax02 = fig.add_subplot(gs[0,2]) # cost vs vacc_eff
    ax10 = fig.add_subplot(gs[1,0]) # mitigation and prevalence example 1
    ax11 = fig.add_subplot(gs[1,1]) # mitigation and prevalence example 2
    ax12 = fig.add_subplot(gs[1,2]) # mitigation and prevalence example 3
    
    # panel a
    # vaccination rate ------------------------------------------------------------------------------------------------------------
    t = np.linspace(0, Tshow, Tshow) # time points for vaccination rate
    print("make sure that vaccination rate is the same function as in SIRS.py!")
    ax00.plot(t, 0.004/(1+np.exp(-0.1*(t-150))), color = 'k') # print vaccination rate
    ax00.set_ylabel('daily new\nvaccinations (%)', color = 'k') # y axis label
    ax00.set_xlabel('time (years)') # x axis label
    ax00.set_ylim(-0.00005,0.0041) # y axis limits
    ax00.get_yaxis().set_ticks([0,0.004]) # y axis ticks
    ax00.set_yticklabels([0,0.4]) # y axis tick labels
    ax00.set_xlim(0,Tshow) # x axis limits
    ax00.get_xaxis().set_ticks(labels*365) # x axis ticks
    ax00.set_xticklabels(labels) # x axis tick labels
    ax00.tick_params(labelbottom=True) # show x axis ticks
    ax00.text(-0.55,1.2,'a',fontsize=8, weight = 'bold',transform=ax00.transAxes) # panel label
    sns.despine(ax=ax00, top=True, right=True, left=False, bottom=False, offset=0, trim=True) # remove top and right spines

    # panel b, d-f
    # susceptible, prevalence and mitigation --------------------------------------------------------------------------------------------
    Cs = [] # array to save costs
    for vacc_eff, ax, abc in zip([0.3,0.5,0.9], [ax10, ax11, ax12],['d','e','f']): # loop over different vaccination efficiencies
        with open('%sa%g_beta1_%g_vacceff%g_nuv%g_data.npy' %(path_known,a,beta_0_1,vacc_eff,nuv), 'rb') as g: # load data
            t = np.load(g)
            S = np.load(g)
            Sv = np.load(g)
            I = np.load(g)
            Iv = np.load(g)
            M = np.load(g)
        I = np.add(I,Iv)
        CM = np.sum(CM_div(M))*dt/len(t) 
        CI = np.sum((CI_cost(a,I)))*dt/len(t)
        Cs.append([CM,CI])

        
        t = t[datafrom:datato]
        I = I[datafrom:datato]
        M = M[datafrom:datato]
        S = S[datafrom:datato]
        Sv = Sv[datafrom:datato]

        #axR = ax02
        ax2 = ax.twinx()
        ax.text(0.07,0.8,r'$\eta$ = %g'%vacc_eff,fontsize=8,transform=ax.transAxes)
        #ax02.text(0.71,0.4,r'$\eta$ = %g'%vacc_eff,fontsize=8,transform=ax02.transAxes)
        ax.set_xlim(0,Tshow)
        ax.get_xaxis().set_ticks(labels*365)
        ax.set_xticklabels(labels)
        ax.set_ylim(0,1)
        ax.set_yticks([0,1])
        ax2.set_ylim(0,0.003)
        yticks = [0,0.003]
        ax2.set_yticks(yticks) 
        ax2.set_yticklabels([0,0.3])
        
        ax2.plot(t, I, label='I', color = 'indianred')
        ax.plot(t, M, label='M', color = 'cornflowerblue')
        #if vacc_eff == 0.3:
        ax.set_ylabel('mitigation', color = 'cornflowerblue')
        #if vacc_eff == 0.3:
        ax2.set_ylabel('prevalence (%)', color = 'indianred', rotation = 270, labelpad = 10)
        if vacc_eff == 0.5:
            ax.set_xlabel('time (years)')
        sns.despine(ax=ax, top=True, right=False, left=False, bottom=False, offset=0, trim=True)
        sns.despine(ax=ax2, top=True, right=False, left=False, bottom=False, offset=0, trim=True)
        for label in ax2.get_yticklabels():
            label.set_color('indianred')
        for label in ax.get_yticklabels():
            label.set_color('cornflowerblue')
        ax.text(-0.35,1.2,abc,fontsize=8, weight = 'bold',transform=ax.transAxes)
    
        if vacc_eff == 0.9:
            axR = ax01
            axR.plot(t, S, color = 'darkolivegreen', label = 'unvaccinated')
            axR.plot(t, Sv, color = 'darkkhaki', label = 'vaccinated')
            axR.set_ylim(-0.01,1.01)
            axR.set_yticks([0,0.5,1])
            axR.set_yticklabels([0,50,100])
            axR.set_xlim(0,Tshow)
            axR.get_xaxis().set_ticks(labels*365)
            axR.set_xticklabels(labels)
            axR.tick_params(labelbottom=True)
            
            ax2.tick_params(labelbottom=True) 
            axR.set_xlabel('time (years)')
            axR.set_ylabel('susceptible (%)', color = 'k')
            axR.text(0.31,0.8,'unprotected', color = 'darkolivegreen' ,fontsize=8, transform=axR.transAxes)
            axR.text(0.31,0.14,'protected', color = 'darkkhaki' ,fontsize=8, transform=axR.transAxes)
            sns.despine(ax=axR, top=True, right=True, left=False, bottom=False, offset=0, trim=True) 
            axR.text(-0.55,1.2,'b',fontsize=8, weight = 'bold',transform=axR.transAxes)
    #axR.legend(loc='lower left', bbox_to_anchor=(0.7, 0.4), fontsize=7, frameon=False)
 
    
    # panel c
    # cost vs vaccination efficiency --------------------------------------------------------------------------------------------

    # array to save costs
    costtots = []
    CMs = []
    CIs = []

    for vacc_eff_ in np.arange(0,1.01,0.05): # loop over different vaccination efficacies
        # load data
        with open('%sa%g_beta1_%g_vacceff%g_nuv%g_data.npy' %(path_known,a,beta_0_1,vacc_eff_,nuv), 'rb') as g: 
            t = np.load(g)
            S = np.load(g)
            Sv = np.load(g)
            I = np.load(g)
            Iv = np.load(g)
            M = np.load(g)
        I = np.add(I,Iv) # prevalence is sum of infected and vaccinated infected
        # calc cost
        CM = CM_div(M)
        CI = CI_cost(a,I)
        CM = np.sum(CM)*dt
        CI = np.sum(CI)*dt
        costtot = (CI + CM)/len(t) # total daily cost
        costtots.append(costtot)
        CMs.append(CM/len(t))
        CIs.append(CI/len(t))
        print(vacc_eff_, costtot)
    
    # plot cost vs vaccination efficiency
    ax = ax02
    ax.plot(np.arange(0,1.01,0.05),costtots, color='k', label = 'total cost')
    ax.plot(np.arange(0,1.01,0.05),CMs, color='cornflowerblue', label = 'mitig. cost')
    ax.plot(np.arange(0,1.01,0.05),CIs, color='indianred', label = 'infect. cost')

    # plot points for different vaccination efficacies (example values from other panels)
    for vacc_eff, Cs, abc in zip([0.3,0.5,0.9],Cs,['d','e','f']):
        CM = Cs[0]
        CI = Cs[1]
        ax.scatter(vacc_eff, CM, color='cornflowerblue', s=8)
        ax.scatter(vacc_eff, CI, color='indianred', s=8)
        ax.scatter(vacc_eff, CM+CI, color='k', s=8)
        ax.text(vacc_eff, CM+CI+0.07,abc, color = 'k' ,fontsize=8)
    # labels etc.
    ax.set_ylabel('mean daily\n total cost (a.u.)', color = 'k')
    ax.set_xlabel(r'vaccine effectiveness $\eta$')
    ax.set_xlim(0,1)
    mincost = 0
    maxcost = 1
    ax.set_ylim(mincost,maxcost)
    ax.get_xaxis().set_ticks([0,0.5,1])
    ax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
    ax.get_yaxis().set_major_formatter(FormatStrFormatter('%g'))
    ax.get_yaxis().set_ticks([mincost,(mincost+maxcost)/2,maxcost])
    ax.text(-0.55,1.2,'c',fontsize=8, weight = 'bold',transform=ax.transAxes)
    ax.legend(loc='lower left', bbox_to_anchor=(0.5,0.4), fontsize=8, frameon=False, handlelength = 1)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, offset=0, trim=True)

    # align y labels
    fig.align_ylabels([ax00, ax01, ax02])
    
    # save figure
    fig.savefig('%splot_vacc_%s_%gyr_a%g_beta1_%g_nuv%g.pdf' %(path0,whichcost,yearstot,a,beta_0_1,nuv), bbox_inches="tight", dpi = 600)




# call plotting function
plot_vacc(path0, a, beta_0_1,nuv)