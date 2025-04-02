# This code runs simulations for determining the optimal mitigation strategy for a given cost function and seasonality
# The code also plots the results of the simulations

# import libraries
import os
os.environ["JAX_LOG_COMPILES"] = "0"
import numpy as np
import diffrax
import jax.numpy as jnp
import jax
import jaxopt
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns
import csv
import matplotlib.gridspec as gridspec
import library_costmin.SIRS as SIRS
import icomo # https://icomo.readthedocs.io/en/stable/


# parameters
whichcost = 'div' # which mitigation cost function, options: 'div', 'divlog', 'divsqrt'
beta_0 = 0.2 # basic reproduction number
phi = 0 # phase shift
gamma = 0.1 # recovery rate
nu = 1/100 # death rate
delta = 0 # delay for measures
I0 = 1e-4 # starting infectious population
a = 10 # disease severity 
T = 10 # simulation time in years
period = 1 # period of seasonality in years
T = T*365 # simulation time in days
delta_beta = 0.1 # amplitude of seasonality


# set plotting parameters
plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('figure', titlesize=10)

# adjust lightness of color
# adapted from https://gist.github.com/ihincks/6a420b599f43fcd7dbd79d56798c4e5a
def adjust_lightness(color, amount): 
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


# infection cost
slope = 5
def CI_cost(a,I):
	h = 0.5*(5+jnp.sqrt(21)) 
	k = 0.001288575763893496
	return a * (1/h*I+h*(I-0.001)*(1-1/(1+jnp.exp(1000*(I-0.001))))+k)



# different CM functions
@jax.jit
def CM_div(m):
    if whichcost  == "div":
        return	m/(1-m) # divergent cost 
    if whichcost == "divlog":
        return -jnp.log(1-m) # logarithmic cost
    if whichcost == "divsqrt":
        return (2/jnp.sqrt(1-m)-2) # square root cost


# seasonality with mean beta_0, apmlitude delta_beta, phaseshift phi
def seasonality(beta_0, delta_beta, phi, t, M,period):
    return (beta_0+delta_beta*jnp.sin(2*jnp.pi*t/(365*period)+phi))*(1-M)


# general seasonality plot
def plot_seas(T, t, I_results_seas, M_results_seas, I_seas_nomit, M_seas_nomit, I_seas_constmit, M_seas_constmit, beta_0, delta_beta, phi,path0,period):
    
    # prepapration -------------------------------------------------------------------------------------------------
    yr = 3 # for how many years to plot

    # optimal mitigation 
    M_plot_seas = M_results_seas[0][3*365:(3+yr)*365] # results for chosen years for optimal mitigation
    I_plot_seas = I_results_seas[0][3*365:(3+yr)*365] # results for chosen years for infections under optimal mitigation
    CM_oneyear_opt = np.sum(CM_div(M_plot_seas[365:2*365])) # mitigation cost per year for optimal mitigation
    CI_oneyear_opt = np.sum(CI_cost(a,I_plot_seas[365:2*365])) # infection cost per year for optimal mitigation
    # print infromation
    print("mit. cost per year optimal:", CM_oneyear_opt) 
    print("cost per year optimal:", CM_oneyear_opt+CI_oneyear_opt)
    print("max I", np.max(I_plot_seas))

    # comparison case: no mitigation 
    M_seas_nomit = M_seas_nomit[3*365:(3+yr)*365] # results for chosen years for no mitigation
    I_seas_nomit = I_seas_nomit[3*365:(3+yr)*365] # results for chosen years for infections under no mitigation
    CM_oneyear_nomit = np.sum(CM_div(M_seas_nomit[365:2*365])) # mitigation cost per year for no mitigation
    CI_oneyear_nomit = np.sum(CI_cost(a,I_seas_nomit[365:2*365])) # infection cost per year for no mitigation
    # print information
    print("mit. cost per year no mit:", CM_oneyear_nomit)
    print("inf. cost per year no mit:", CI_oneyear_nomit)
    print("inf. for no mitigation:", I_seas_nomit[365:2*365])
    print("inf. cost for no mitigation:", CI_cost(a,I_seas_nomit[365:2*365]))
    print("cost per year no mit:", CM_oneyear_nomit+CI_oneyear_nomit)
    print("max I no mit", np.max(I_seas_nomit))

    # comparison case: constant mitigtaion at the mean of the optimal mitigation
    M_seas_constmit = M_seas_constmit[3*365:(3+yr)*365] # results for chosen years for constant mitigation
    I_seas_constmit = I_seas_constmit[3*365:(3+yr)*365] # results for chosen years for infections under constant mitigation
    CM_oneyear_constmit = np.sum(CM_div(M_seas_constmit[365:2*365])) # mitigation cost per year for constant mitigation
    CI_oneyear_constmit = np.sum(CI_cost(a,I_seas_constmit[365:2*365])) # infection cost per year for constant mitigation
    # print information
    print("cost per year const mit:", CM_oneyear_constmit+CI_oneyear_constmit)
    print("ratio:", np.max(I_plot_seas)/np.max(I_seas_nomit))
    print("max I no mit", np.max(I_seas_nomit))
    print("max I const mit", np.max(I_seas_constmit))
    print("max I opt mit", np.max(I_plot_seas))

    # time for plotting
    t_plot = t[0:yr*365]
    
    # create figure and add panels
    fig = plt.figure(figsize=(7/2.54, 15/2.54))
    gs_ = gridspec.GridSpec(2,1, height_ratios=[4,1],hspace = 0.3, figure = fig)
    gs = gs_[0].subgridspec(4,1, height_ratios=[1,1,1,1], hspace=0.4)
    gs1 = gs_[1].subgridspec(1,1)
    ax0 = fig.add_subplot(gs[0]) 
    ax2 = fig.add_subplot(gs[1]) 
    ax1 = fig.add_subplot(gs[2]) 
    ax3 = fig.add_subplot(gs[3]) 
    ax4 = fig.add_subplot(gs1[0]) 
    
    # calculate seasonality
    seas = seasonality(beta_0, delta_beta, phi, t, np.zeros(len(t)),period)/gamma
    

    # bar plot for costs --------------------------------------------------------------------------------------------------------------------------------
    Ctots = [CM_oneyear_nomit+CI_oneyear_nomit, CM_oneyear_constmit+CI_oneyear_constmit, CM_oneyear_opt+CI_oneyear_opt] # array with total costs for all three cases
    CMs = [CM_oneyear_nomit, CM_oneyear_constmit, CM_oneyear_opt] # array with mitigation costs for all three cases
    CIs = [CI_oneyear_nomit, CI_oneyear_constmit, CI_oneyear_opt] # array with infection costs for all three cases
    barWidth = 0.25 # width of the bars
    br1 = np.arange(len(Ctots)) # x values for bar plot for total costs
    br2 = [x + barWidth for x in br1]  # x values for bar plot for mitigation costs
    br3 = [x + barWidth for x in br2]  # x values for bar plot for infection costs
    # plot bar plot
    ax4.bar(br1, np.array(Ctots)/365, color ='k', width = barWidth*0.8, label ='total cost') 
    ax4.bar(br2, np.array(CMs)/365, color ='cornflowerblue', width = barWidth*0.8, label ='mit. cost') 
    ax4.bar(br3, np.array(CIs)/365, color ='indianred', width = barWidth*0.8, label ='inf. cost') 
    # add labels and ticks
    ax4.set_ylabel('mean daily\ncosts (a.u.)') 
    ax4.set_xticks([r + barWidth for r in range(len(Ctots))], [r'no mitigation', r'constant', r'optimal', ])
    ax4.set_ylim([0,2])
    ax4.set_yticks([0,1,2])
    # cosmetics
    ax4.legend(loc='upper right', bbox_to_anchor=(1.35, 1.2), facecolor='white', framealpha=0.8, edgecolor='white') # add legend
    sns.despine(ax=ax4, top=True, right=True, left=False, bottom=False, offset=0, trim=False) # remove spines
    

    # measure phase shift
    Mseas = M_results_seas[0]
    Iseas = I_results_seas[0]
    measurefrom = 365*5 # meaure in year six to exclude transient effects
    maxM_index =  np.argmax(Mseas[int(measurefrom):int(measurefrom+365*period)]) # index of maximum optimal mitigation
    maxM = np.max(Mseas[int(measurefrom):int(measurefrom+365*period)]) # maximum optimal mitigation
    maxI_index =  np.argmax(Iseas[measurefrom+maxM_index:int(measurefrom+maxM_index+365*period)]) # index of maximum infections measured from maximum mitigation index
    maxI = np.max(Iseas[measurefrom+maxM_index:int(measurefrom+maxM_index+365*period)]) # maximum infections
    phaseshift = (maxI_index)/(365*period)*2*np.pi # phase shift in radians
    # write phase shift to file
    file = open(path0 + '%s/seasonality_%s_%.3fyrperiod_T%s_a%g.dat' %(whichcost,whichcost,period,T,a), 'w')	
    file.write('%f,%f,%f,%f,%f,%f' %(maxM_index,maxM,maxI_index,maxI,phaseshift,period))

    
    
    # main plots ------------------------------------------------------------------------------------------------------------------
    # background for seasons
    for p in np.arange(0, 365*yr, 365):
        ax0.axvline(x=p+365/4, ymin=-1.3, ymax=0.65, color='whitesmoke', zorder = -1, clip_on=False, linewidth = 30)
        ax1.axvline(x=p+365/4, ymin=-1, ymax=0.9, color='whitesmoke', zorder = -1, clip_on=False, linewidth = 30)
        ax2.axvline(x=p+365/4, ymin=-1, ymax=0.9, color='whitesmoke', zorder = -1, clip_on=False, linewidth = 30)
        ax3.axvline(x=p+365/4, ymin=0.3, ymax=0.9, color='whitesmoke', zorder = -1, clip_on=False, linewidth = 30)
    

    # plot basic reproduction number to show seasonality
    ax0.axhline(y=1, xmin=0, xmax=1, color='lightgrey', clip_on=False) # add line at R0 = 1
    ax0.plot(t, seas, color = 'dimgrey') # plot seasonality
    ax0.set_ylabel('basic\nreprod.\nnumber', color = 'dimgrey') # add ylabel
    # set ticks and limits
    ax0.set_ylim([0,4.05])
    ax0.get_yaxis().set_ticks([0,1,2,3,4])
    ax0.set_xlim(0, yr*365)
    ax0.get_xaxis().set_ticks(np.arange(0, yr*365+1, 365))
    ax0.tick_params(labelbottom=False) 
    # add winter and summer labels
    ax0.text(0.02,1.05,'winter',transform = ax0.transAxes, color = 'dimgrey', fontsize = 8)
    ax0.text(0.16,0.1,'summer',transform = ax0.transAxes, color = 'dimgrey', fontsize = 8)
    sns.despine(ax=ax0, top=True, right=True, left=False, bottom=False, offset=0, trim=True) # remove spines
    
    
    # plot optimal mitigation and infections
    axi1=ax1.twinx() # add second y-axis for mitigation
    axi1.plot(t_plot, M_plot_seas, color = 'cornflowerblue') # plot optimal mitigation
    ax1.plot(t_plot, I_plot_seas, color = 'indianred') # plot infections under optimal mitigation
    axi1.set_ylabel('optimal\nmitigation\n'+r'$M_{opt}$', color = 'cornflowerblue', rotation = -90) # add ylabel for mitigation
    ax1.set_ylabel('prevalence (%)', color = 'indianred') # add ylabel for infections
    # set ticks and limits
    axi1.set_ylim([0,1])
    axi1.set_yticks([0,1])
    ax1.set_ylim(0,0.003)
    ax1.set_yticks([0,0.003])
    ax1.set_yticklabels([0,0.3])
    axi1.set_xlim(0, yr*365)
    ax1.set_xlim(0, yr*365)
    axi1.get_xaxis().set_ticks(np.arange(0, yr*365+1, 365))
    ax1.tick_params(labelbottom=False) 
    sns.despine(ax=axi1, top=True, right=False, left=False, bottom=False, offset=0, trim=False) # remove spines
    sns.despine(ax=ax1, top=True, right=False, left=False, bottom=False, offset=0, trim=False) # remove spines
    # for the default case (divergent cost fct) add phase shift visualization to plot
    if a == 10 and beta_0 == 0.2 and delta_beta == 0.1 and whichcost == 'div':
        axi1.axvline(x=int(365+maxM_index), ymin = float(maxM), ymax=0.81, color='k', clip_on=False,zorder = 0, linewidth = 1)
        axi1.axvline(x=int(365+maxM_index+maxI_index), ymin=0.35, ymax=0.81, color='k', clip_on=False, linewidth = 1)
        axi1.annotate('', xy=(int(365+maxM_index)-10, 0.8), xycoords='data',xytext=(int(365+maxM_index+maxI_index)+10, 0.8), textcoords='data',arrowprops=dict(arrowstyle = '<->'))
        axi1.annotate(r'$1/4$ year', xy=(int(365+maxM_index)-40, 0.95), xycoords='data', color = 'k', fontsize = 8)
        axi1.scatter(365+maxM_index,maxM, color = 'cornflowerblue', s = 10)
        ax1.scatter(365+maxM_index+maxI_index,maxI, color = 'indianred', s = 10,zorder = 1000)
    # make sure the y-axis labels are in the right color
    for label in ax1.get_yticklabels():
        label.set_color('indianred')
    for label in axi1.get_yticklabels():
        label.set_color('cornflowerblue')


    # plot effective reproduction number for optimal mitigation ----------------------------------------------------------------------------------------------
    axi2 = ax2 
    axi2.axhline(y=1, xmin=0, xmax=1, color='lightgrey', clip_on=False) # add line at Reff = 1
    R = seasonality(beta_0, delta_beta, phi, t_plot, M_plot_seas,period)/gamma # calculate effective reproduction number from basic repr. number and optimal mitigation
    axi2.plot(t_plot,R, color = 'k') # plot effective reproduction number
    print("Reff max and min:", np.max(R), np.min(R)) # print max and min effective reproduction number
    axi2.set_ylabel('effective\nreprod.\nnumber', color = 'k') # add ylabel
    # set ticks and limits
    axi2.set_xlim(0, yr*365)
    axi2.set_ylim([0,4])
    axi2.set_yticks([0,1,2,3,4])
    axi2.get_xaxis().set_ticks(np.arange(0, yr*365+1, 365))
    axi2.spines['top'].set_visible(False) # remove spines
    axi2.spines['right'].set_visible(False) # remove spines
    axi2.tick_params(labelbottom=False) 


    # plot different mitigation strategies in one plot ------------------------------------------------------------------------------------------------------
    axi3 = ax3
    # no mitigation
    axi3.plot(t_plot, I_seas_nomit, color = adjust_lightness('indianred', 1.4), label = r'no mitigation')
    # set ticks and limits etc.
    axi3.set_ylim(-0.01,0.2)
    axi3.set_yticks([0,0.2])
    axi3.set_yticklabels([0,20])
    axi3.set_ylabel('prevalence (%)', color = 'indianred')
    axi3.set_xlim(0, yr*365)
    axi3.set_xticks(np.arange(0, yr*365+1, 365))
    axi3.set_xticklabels(np.arange(0, yr+1, 1))
    axi3.set_xlabel('time (years)')
    fig.align_ylabels([axi3,ax1])
    sns.despine(ax=axi3, top=True, right=True, left=False, bottom=False, offset=0, trim=True)
    # const mitigation
    axi3.plot(t_plot, I_seas_constmit, color = adjust_lightness('indianred', 0.5), label = r'constant')
    # optimal mitigation
    axi3.plot(t_plot, I_plot_seas, color = 'indianred', label = r'optimal')
    # add legend
    axi3.legend(loc='upper right', bbox_to_anchor=(1.35, 1.2), facecolor='white', framealpha=0.8, edgecolor='white')
    for label in axi3.get_yticklabels():
        label.set_color('indianred')

    # add panel labels
    ax0.text(-0.3,0.9,'a',fontsize=8, weight = 'bold', transform=ax0.transAxes)
    axi1.text(-0.3,0.9,'c',fontsize=8, weight = 'bold', transform=axi1.transAxes)
    axi2.text(-0.3,0.9,'b',fontsize=8, weight = 'bold', transform=axi2.transAxes)
    axi3.text(-0.3,0.9,'d',fontsize=8, weight = 'bold', transform=axi3.transAxes)
    ax4.text(-0.3,0.9,'e',fontsize=8, weight = 'bold', transform=ax4.transAxes)
    
    # save figure
    fig.show()
    fig.savefig('./data/seasonality_%s/seasonality_%s_T%s_a%g_beta0%g_deltabeta%g.pdf' %(whichcost,whichcost,T,a,beta_0,delta_beta), bbox_inches="tight") 
    plt.close()


# run simulation -----------------------------------------------------------------------------------------------------------------------------------------

# arrays for results
I_results = [[],[],[]]
M_results = [[],[],[]]
Ctot_results = [[],[],[]]
I_results_seas = [[],[],[]]
M_results_seas = [[],[],[]]
Ctot_results_seas = [[],[],[]]
  
# path for saving files
date = 'seasonality_beta0%g_deltabeta%g_a%g' %(beta_0,delta_beta,a)
path0 = './data/seasonality_%s/'%whichcost + date + '/' 
for costpath in [whichcost]:
    path = path0 + costpath + '/'
    if not os.path.exists(path):
        #print("path for saving files does not exist, creating new directory")
        os.makedirs(path)
        #print("new directory created: %s" %path)

# simulation preparation
len_sim = T
num_points = T
dt = len_sim/(num_points-1)
t_out = np.linspace(0, len_sim, num_points)
t_beta = np.linspace(0, len_sim, num_points//1)

# arrays to save results
I_results_ = [[],[],[],[]]
M_results_ = [[],[],[],[]]
Ctot_results_ = [[],[],[],[]]
I_results_seas_ = [[],[],[],[]]
M_results_seas_ = [[],[],[],[]]
Ctot_results_seas_ = [[],[],[],[]]

# path for saving files
path = path0 + whichcost + '/'
CM_func = CM_div
CM_name = whichcost
CI_func = CI_cost

# without seasonality -----------------------------------------------------------------------------------------------------
seas = False
try: # try to load old data
    with open('%s%.3fyrperiod_T%s_CM%s_a%g_seas%s_data000.npy' %(path,period,T,CM_name,a,seas), 'rb') as g: 
        t = np.load(g)
        S = np.load(g)
        I = np.load(g)
        M = np.load(g)
    print('loaded old data instead of new simulation')
except:
    print('starting sim without seasonality')

    # simulation function
    def simulation(x):
        frac_reduc = jax.nn.sigmoid(-x)*0.99999999999999999 # sigmoid function to get values between 0 and 1
        beta_t = frac_reduc*beta_0 # calculate beta_t
        beta_t_func = icomo.interpolate_func(ts_in=t_beta,values=beta_t) # beta is now callable function
        y0 = {'S': 1-I0, 'I': I0, 'R':0} # initial conditions
        args = {'gamma': gamma, 'nu': nu, 'beta_t': beta_t_func} # arguments for ODE
        
        output = icomo.diffeqsolve(args=args, ODE=SIRS.SIRS, y0=y0, ts_out=t_out) # solve ODE

        eff_frac_reduc = icomo.interpolate_func(t_beta, frac_reduc, 'cubic')(t_out) # interpolate frac_reduc to t_out
        return output.ys, eff_frac_reduc
        
    # cost function
    @jax.jit
    def min_func(x):
        output, frac_reduc = simulation(x)
        m = 1-frac_reduc
        cost = jnp.sum(CI_func(a,output['I']) + CM_func(m))*dt
        return cost
    
    # initial values for optimization
    M0 = 0.5
    x_0_ = np.log(1/(1-M0)-1)
    x_0 = np.ones(len(t_beta))*x_0_
    
    # use automatic differentiation to calculate gradients
    value_and_grad_min_func = jax.jit(jax.value_and_grad(min_func))

    # use L-BFGS-B to optimize the cost function
    solver = jaxopt.ScipyMinimize(fun=value_and_grad_min_func, value_and_grad=True, method="L-BFGS-B", jit=False, maxiter=1000)

    # run optimization
    res = solver.run(x_0)
            
    # results
    output, frac_reduc = simulation(res.params)
    mitigation = 1-frac_reduc

    # save information about simulation
    f = open('%s%.3fyrperiod_T%s_CM%s_a%g_seas%s_info.dat' %(path,period,T,CM_name,a,seas), 'w')	
    f.write("beta_0 %f \n gamma %f \n nu %f \n a %f \n I0 %f \n M0 %f \n delta %f \n len_sim %f \n num_points %f \n CM_name %s \n res.state.fun_val %f \n " %(beta_0,gamma,nu,a,I0,M0,delta,len_sim,num_points,CM_name,res.state.fun_val))

    # save results
    I = output['I']
    M = mitigation
    t = t_out
    with open('%s%.3fyrperiod_T%s_CM%s_a%g_seas%s_data.npy' %(path,period,T,CM_name,a,seas), 'wb') as f:
        np.save(f, t)
        np.save(f, output['S'])
        np.save(f, I)
        np.save(f, M)

# with seasonality ---------------------------------------------------------------------------------------------------
seas = True
try: # try to load old data
    with open('%s%.3fyrperiod_T%s_CM%s_a%g_seas%s_data000.npy' %(path,period,T,CM_name,a,seas), 'rb') as g: # CHANGE
        t = np.load(g)
        S_seas = np.load(g)
        I_seas = np.load(g)
        M_seas = np.load(g)
    print('loaded old data instead of new simulation')

except:
    print('starting sim with seasonality')

    # simulation function for seasonality
    def simulation_seas(x):
        frac_reduc = jax.nn.sigmoid(-x)*0.99999999999999999 # sigmoid function to get values between 0 and 1
        frac_reduc_func = icomo.interpolate_func(ts_in=t_beta,values=frac_reduc) # beta is now callable function
        y0 = {'S': 1-I0, 'I': I0, 'R':0} # initial conditions
        args = {'gamma': gamma, 'nu': nu, 'beta_0': beta_0, 'delta_beta': delta_beta, 'phi': phi, 'frac_reduc': frac_reduc_func} # arguments for ODE
        
        output = icomo.diffeqsolve(args=args, ODE=SIRS.SIRS_seas, y0=y0, ts_out=t_out) # solve ODE

        eff_frac_reduc = icomo.interpolate_func(t_beta, frac_reduc, 'cubic')(t_out) # interpolate frac_reduc to t_out
        return output.ys, eff_frac_reduc
    
    # simulation function for seasonality with constant mitigation
    def simulation_seas_constmit(length,M_const):
        frac_reduc = np.ones(length)*(1-M_const) # calculate beta_t
        frac_reduc_func = icomo.interpolate_func(ts_in=t_beta,values=frac_reduc) # beta is now callable function
        y0 = {'S': 1-I0, 'I': I0, 'R':0} # initial conditions
        args = {'gamma': gamma, 'nu': nu, 'beta_0': beta_0, 'delta_beta': delta_beta, 'phi': phi, 'frac_reduc': frac_reduc_func} # arguments for ODE
        
        output = icomo.diffeqsolve(args=args, ODE=SIRS.SIRS_seas, y0=y0, ts_out=t_out) # solve ODE

        eff_frac_reduc = icomo.interpolate_func(t_beta, frac_reduc, 'cubic')(t_out) # interpolate frac_reduc to t_out
        return output.ys, eff_frac_reduc
        
    # cost function for seasonality
    @jax.jit
    def min_func_seas(x):
        output, frac_reduc = simulation_seas(x)
        m = 1-frac_reduc
        cost = jnp.sum(CI_func(a,output['I']) + CM_func(m))*dt
        return cost

    # initial values for optimization
    M0 = 0.5 # initial mitigation
    x_0_ = np.log(1/(1-M0)-1) # initial value for optimization
    x_0 = np.ones(len(t_beta))*x_0_ # initial values for optimization in array

    # use automatic differentiation to calculate gradients
    value_and_grad_min_func_seas = jax.jit(jax.value_and_grad(min_func_seas))

    # use L-BFGS-B to optimize the cost function
    solver_seas = jaxopt.ScipyMinimize(fun=value_and_grad_min_func_seas, value_and_grad=True, method="L-BFGS-B", jit=False, maxiter=1000)

    # run optimization
    res_seas = solver_seas.run(x_0)

    # results   
    output_seas, frac_reduc_seas = simulation_seas(res_seas.params)
    mitigation_seas = 1-frac_reduc_seas
    length = len(t_out)
    output_seas_nomit, frac_reduc_seas_nomit = simulation_seas_constmit(length,0) 
    mitigation_seas_nomit = 1-frac_reduc_seas_nomit
    output_seas_constmit, frac_reduc_seas_constmit = simulation_seas_constmit(length,np.mean(mitigation_seas[3*365:(6)*365]))
    mitigation_seas_constmit = 1-frac_reduc_seas_constmit

    # save information about simulation
    f = open('%s%.3fyrperiod_T%s_CM%s_a%g_seas%s_info.dat' %(path,period,T,CM_name,a,seas), 'w')	
    f.write("beta_0 %f \n delta_beta %f \n phi %f \n gamma %f \n nu %f \n a %f \n I0 %f \n M0 %f \n delta %f \n len_sim %f \n num_points %f \n CM_name %s \n res.state.fun_val %f \n " %(beta_0,delta_beta,phi,gamma,nu,a,I0,M0,delta,len_sim,num_points,CM_name,res_seas.state.fun_val))

    # save results
    I_seas = output_seas['I']
    M_seas = mitigation_seas
    t = t_out
    with open('%s%.3fyrperiod_T%s_CM%s_a%g_seas%s_data.npy' %(path,period,T,CM_name,a,seas), 'wb') as f:
        np.save(f, t)
        np.save(f, output_seas['S'])
        np.save(f, I_seas)
        np.save(f, M_seas)   

# write results to arrays
I_results_[0] = I
M_results_[0] = M
Ctot_results_[0] = jnp.sum(CI_func(a,I) + CM_func(M))*dt
I_results_seas_[0] = I_seas
M_results_seas_[0] = M_seas
Ctot_results_seas_[0] = jnp.sum(CI_func(a,I_seas) + CM_func(M_seas))*dt

# plot results -----------------------------------------------------------------------------------------------------------------------------------------
plot_seas(T, t,  I_results_seas_, M_results_seas_, output_seas_nomit['I'], mitigation_seas_nomit, output_seas_constmit['I'], mitigation_seas_constmit, beta_0, delta_beta, phi,path0,period)

