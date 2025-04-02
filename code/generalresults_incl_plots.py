# This script generates the data and plots for the general results section of the paper


# import libraries
import os
os.environ["JAX_LOG_COMPILES"] = "0"
import numpy as np
import diffrax
import jax.numpy as jnp
import jax
import jaxopt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import icomo # https://icomo.readthedocs.io/en/stable/

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

whichcost = 'div' # which mitigation cost to use, options: 'div', 'divlog', 'divsqrt'
beta_0 = 0.2 # basic transmission rate, R_0 = beta_0/gamma
gamma = 0.1 # recovery rate
nu = 1/100 # rate of immunity loss
I0 = 1e-4 # initial infectious population
bs = [0.01, 0.1, 1] # severity of disease, a = 1/b, three values that we used for example plots
longbs = np.sort(np.concatenate([np.arange(0.23,0.27,0.02),np.arange(0.01,1,0.01)])) # severity of disease, a = 1/b, range of values for the plot where M and I are plotted against b
years = 5 # number of years to simulate
T = years*365 # number of days to simulate

date = 'general_T%g_beta0%g_b%g_%g_%g' %(years,beta_0,bs[0],bs[1],bs[2]) # path for saving files
path0 = './data/general_%s/'%whichcost + date + '/'  # path for saving files
path = path0 + whichcost + '/' # path for saving files
for costpath in [whichcost]: # create directories if they do not exist
    path = path0 + costpath + '/'
    if not os.path.exists(path):
        #print("path for saving files does not exist, creating new directory")
        os.makedirs(path)
        #print("new directory created: %s" %path)

# set plot parameters
plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('figure', titlesize=10)

# infection cost
def CI_cost(I):
	a = 0.5*(5+jnp.sqrt(21)) # slope 5
	b = 0.001288575763893496
	return 1/a*I+a*(I-0.001)*(1-1/(1+jnp.exp(1000*(I-0.001))))+b



# mitigation cost
def CM_div(b,m):
    if whichcost  == "div":
        return	b * m/(1-m) # divergent cost 
    if whichcost == "divlog":
        return -b*jnp.log(1-m) # divergent log cost
    if whichcost == "divsqrt":
        return b * (2/jnp.sqrt(1-m)-2) # divergent sqrt cost

# equilibrium infectious population for a given mitigation level
def I_star(M):
    if jnp.isscalar(M): # check if array or scalar
        if beta_0*(1-M) >= gamma:
            return - (nu* (gamma/(beta_0*(1-M)) - 1)) / (gamma + nu)
        else:
            return 0
    else:
        return np.array([I_star(Mi) for Mi in M])

# total cost
def Ctot(b,M):
    return CM_div(b,M)+CI_cost(I_star(M))

#-----------------------------------------------------------------------------------------------------------------
# which cost to use
CM_func = CM_div
CM_name = whichcost
CI_func = CI_cost

# parameters
len_sim = T # length of simulation
num_points = T # number of points in simulation
dt = len_sim/(num_points-1) # time step
t_out = np.linspace(0, len_sim, num_points) # time points for output
t_solve_ODE = np.linspace(0, len_sim, num_points//1) # time points for solving ODE
t_beta = np.linspace(0, len_sim, num_points//1) # time points for beta


# create data
for b in longbs: # loop over values of b
    break
    integrator = icomo.ODEIntegrator( # for normal SIRS for [0,T]
        ts_out=t_out,
        t_0=t_solve_ODE[0],
        t_1=t_solve_ODE[-1],
        ts_solver=t_solve_ODE,
        ts_arg=t_beta,
        interp="cubic",
        solver=diffrax.Bosh3(),  # a 3rd order method
        #adjoint=diffrax.BacksolveAdjoint(),
        adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=len(t_solve_ODE)),
    )

    def SIRS(t, y, args):
        β, const_arg = args
        γ = const_arg["gamma"]
        nu = const_arg["nu"]
        dS = -β(t) * y["I"] * y["S"] + nu * y["R"]
        dI = β(t) * y["I"] * y["S"]- γ * y["I"]
        dR = γ * y["I"] - nu * y["R"]
        dy = {"S": dS, "I": dI, "R": dR}
        return dy
    SIRS_integrator = integrator.get_func(SIRS)


    def simulation(x):
        frac_reduc = jax.nn.sigmoid(-x)*0.99999999999999999
        beta_t = frac_reduc*beta_0
        y0 = {'S': 1-I0, 'I': I0, 'R':0}
        const_args = {'gamma': gamma, 'nu': nu}
        
        output = jax.jit(SIRS_integrator)(
            y0=y0, arg_t=beta_t, constant_args=const_args
        )

        eff_frac_reduc = icomo.interpolation_func(t_beta, frac_reduc, 'cubic').evaluate(t_out)
        return output, eff_frac_reduc
        
    @jax.jit
    def min_func(x):
        output, frac_reduc = simulation(x)
        m = 1-frac_reduc
        cost = jnp.sum(CI_func(output['I']) + CM_func(b,m))*dt
        return cost


    M0_ = 0.5
    x_0_ = np.log(1/(1-M0_)-1)
    x_0 = np.ones(len(t_beta))*x_0_
    M0 = 1-jax.nn.sigmoid(-x_0[0])*0.99999999999999999

    value_and_grad_min_func = jax.jit(jax.value_and_grad(min_func))

    solver = jaxopt.ScipyMinimize(fun=value_and_grad_min_func, value_and_grad=True, method="L-BFGS-B", jit=False, maxiter=1000)#tol = 1e-10, 
    res = solver.run(x_0)
            
    output, frac_reduc = simulation(res.params)
    mitigation = 1-frac_reduc

    # CHANGE
    f = open('%sT%s_CM%s_b%g_info.dat' %(path,years,CM_name,b), 'w')	
    f.write("beta_0 %f \n gamma %f \n nu %f \n b %f \n I0 %f \n M0 %f \n delta %f \n len_sim %f \n num_points %f \n CM_name %s \n res.state.fun_val %f \n " %(beta_0,gamma,nu,b,I0,M0,delta,len_sim,num_points,CM_name,res.state.fun_val))

    I = output['I']
    M = mitigation
    t = t_out
    # CHANGE
    with open('%sT%s_CM%s_b%g_data.npy' %(path,years,CM_name,b), 'wb') as f:
        np.save(f, t)
        np.save(f, output['S'])
        np.save(f, I)
        np.save(f, M)


#-----------------------------------------------------------------------------------------------------------------
# plotting

# plot general results for three example values of b without any sceanrios, just normal SIRS model
def plot_general(bs):

    # create figure
    fig, axs = plt.subplots(3, figsize=(5/2.54, 5.5/2.54),gridspec_kw=dict(height_ratios=[2,2,2], hspace=0.3), sharex=True)
    fig.subplots_adjust(hspace=0.4)


    for i,b in enumerate(bs): # go through the three values of b
        ax = axs[i] # select the subplot
        path = path0 + whichcost + '/' # path where data for this b is saved
        with open('%sT%s_CM%s_b%g_data.npy' %(path,years,whichcost,b), 'rb') as g:  # load data
            t = np.load(g)
            S = np.load(g)
            I = np.load(g)
            M = np.load(g)
        print("a: ", 1/b)
        print("M: ", M[:4*365])
        ax2=ax.twinx() # create second y-axis for infectious population
        ax.plot(t[:4*365+1], M[:4*365+1], color = 'cornflowerblue') # plot mitigation
        ax2.plot(t[:4*365+1], I[:4*365+1], color = 'indianred') # plot infectious population

        if i ==1: # only want y-axis label for middle subplot
            ax.set_ylabel('mitigation', color = 'cornflowerblue')
            ax2.set_ylabel(r'infectious population ($\%$)', color = 'indianred')
        # set y-axis limits and ticks   
        ax.set_ylim([-0.01,1])
        ax.set_yticks([0,1])
        ax2.set_ylim(0,0.003)
        ax2.set_yticks([0,0.003])
        ax2.set_yticklabels(['0','0.3'])
        if b == 1: # for b=1 we need different y-axis limits as infections are very high
            ax2.set_ylim(0,0.2)
            ax2.set_yticks([0,0.2]) 
            ax2.set_yticklabels(['0','20'])    
        ax.set_xlim(0, 4*365+1)
        ax.set_xlim(0, 4*365+1)
        ax.get_xaxis().set_ticks(np.arange(0, 4*365+1, 365))
        labels = [0,1,2,3,4]
        ax.set_xticklabels(labels)
        ax.tick_params(labelbottom=False)
        sns.despine(ax=ax, top=True, right=False, left=False, bottom=False, offset=0, trim=True) # despine plot
        sns.despine(ax=ax2, top=True, right=False, left=False, bottom=False, offset=0, trim=True) # despine plot
        ax2.set_xticks(np.arange(0, 4*365+1, 365)) 
        for label in ax.get_yticklabels(): # make ticks the same color as the lines
            label.set_color('cornflowerblue')
        for label in ax2.get_yticklabels():
            label.set_color('indianred')

        # add text to plot about what b is (a=1/b)
        ax.annotate(r'$a=%g$' %(1/b), xy=(0.67, 0.85), xycoords='axes fraction', fontsize=8, ha='left', va='center')

    # add x-axis label only to bottom subplot    
    axs[2].tick_params(labelbottom=True)
    axs[2].set_xlabel('time (years)')

    # save figure
    #fig.savefig('./data/general_%s/'%whichcost + 'generalplot_T%g_beta0%g_b%g_%g_%g.pdf' %(years,beta_0,bs[0],bs[1],bs[2]), bbox_inches="tight") 
    #plt.close()

# plot mitigation and equilibrium infectious population as a function of b
def plot_mitvsb(bs):
     
    # create figure
    fig, ax = plt.subplots(1, figsize=(4.5/2.54, 5.5/2.54))

    # arrays to store data
    #meanMs = [] # mean mitigation from numerical solution
    #totalIs = [] # total infections from numerical solution
    analytMs = [] # mean mitigation from semi-analytical solution
    analytIs = [] # total infections from semi-analytical solution
    #equilibriumIs = [] # equilibrium infectious population

    # values of M to consider
    Ms = np.linspace(0,0.99999,1000)

    for b in bs: # go through the values of b
        path = path0 + whichcost + '/' # path where data is saved
        with open('%sT%s_CM%s_b%g_data.npy' %(path,years,whichcost,b), 'rb') as g: # load data
            t = np.load(g)
            S = np.load(g)
            I = np.load(g)
            M = np.load(g)

        #meanMs.append(np.mean(M[:4*365]))
        #totalIs.append(np.sum(I[:4*365])/10)
        minCtotindex = np.argmin(Ctot(b,Ms)) # check for which M the total cost is minimized assuming equilibrium infectious population
        analytM = Ms[minCtotindex] # get the mitigation level that minimizes the total cost
        analytMs.append(analytM)
        analytI = I_star(analytM) # get the equilibrium infectious population for this mitigation level
        print(1/b, analytM, analytI) # print the values
        analytIs.append(analytI)
        #equilibriumIs.append(I[3*365])

    # plot infections
    ax2 = ax.twinx() # create second y-axis for equilibrium infectious population
    ax2.plot(1/np.array(bs), analytIs, color = 'indianred', zorder = 0) # plot equilibrium infectious population
    ax2.set_ylabel('equilibrium infectious\npopulation (%)', color = 'indianred', rotation = 270, labelpad = 20) # set y-axis label
    # set y-axis limits and ticks
    ax2.set_ylim(0,0.08)
    ax2.set_yticks([0,0.04,0.08])
    ax2.set_yticklabels(['0','4','8'])
    sns.despine(ax=ax2, top=True, right=False, left=False, bottom=False, offset=0, trim=False) # despine plot
    for label in ax2.get_yticklabels(): # make ticks the same color as the lines
        label.set_color('indianred')
    
    # plot mitigtaion
    ax.plot(1/np.array(bs),analytMs, c = 'cornflowerblue', zorder = 1, label = 'analytical solution') # plot mitigation
    ax.set_xlim(1,100) # set x-axis limits
    ax.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
    ax.patch.set_visible(False) # hide the 'canvas'
    ax.set_ylabel('mitigation', color = 'cornflowerblue') # set y-axis label
    # set y-axis limits and ticks
    ax.set_ylim([-0.01,1])
    ax.set_yticks([0,0.5,1])
    ax.set_xscale('log') # set x axis log scale
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g')) 
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, offset=0, trim=False) # despine plot
    for label in ax.get_yticklabels(): # make ticks the same color as the lines
        label.set_color('cornflowerblue')
    
    ax.set_xlabel('disease severity $a$') # set x-axis label
    # save figure
    fig.savefig('./data/general_%s/'%whichcost + 'generalplot_meanMvsb_T%g_beta0%g.pdf' %(years,beta_0), bbox_inches="tight") 
    


# create plots
plot_general(bs)
#plot_mitvsb(longbs)