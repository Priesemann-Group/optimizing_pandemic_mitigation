# Description: This script is used to simulate the SIRS model with two variants and find the optimal mitigation strategy. 
# This file is for the scenario where mitigation adjustment is delayed

# import libraries
import os
os.environ["JAX_LOG_COMPILES"] = "0"
import numpy as np
import diffrax
import jax.numpy as jnp
import jax
import jaxopt
import library_costmin.SIRS as SIRS
import matplotlib.pyplot as plt
import seaborn as sns
import icomo # https://icomo.readthedocs.io/en/stable/

# parameters --------------------------------------------------------------------------------------------------------------------
whichcost = 'div' # which cost function to use, options are 'div', 'divlog', 'divsqrt'
a = 1 # disease severity
years = 9 # years of simulation
T_tot = years*365+90 # 1 year + 90 days for decrease of beta0
beta_0_1 = 0.2 # beta for variant 1
beta_diff_factor = 1.5 # factor for beta_2
beta_0_2 = beta_0_1*beta_diff_factor # beta for variant 2
beta_0_1_t = np.concatenate([np.ones(years*365)*beta_0_1,[beta_0_1 - x*beta_0_1/89 for x in range(90)]]) # beta_0_1 for the first year and then decrease to 0 over 90 days
gamma = 0.1 # recovery rate
nu = 1/100 # waning rate
T_part1 = T_tot # time of first part of simulation (run for whole time so that we dont have boundary effects)
# starting values for variant 1 and 2, given are the best starting values for different cost functions etc so that we dont have long times until equilibrium is reached
#I01 = 0.00083 # for beta=0.2 and a=10
#I01 = 0.00034955773027785253 # for beta=0.2 nd a=10, divlog and divsqrt
I01 = 0.04554786472600939 # for beta=0.2 and a=2
#I01 = 0.0001127868959310304 # for beta=0.2 and a=50
#I01 = 0.0454556834290207 # for beta=0.2 and a=1/0.3
#I01 = 0.001808440551412749 # for beta=0.2 and a=5
I02 = 0.0000001 # starting value for variant 2

# where to save the data
date = 'newvariant_unknown_%gyr_a%g' %(years,a)
path0 = './data/newvariant_multcomp_%s/' %whichcost + date + '/' 
for costpath in [whichcost]: # check if the path exists and create it if not
    path = path0 + costpath + '/'
    if not os.path.exists(path):
        print("path for saving files does not exist, creating new directory")
        os.makedirs(path)
        print("new directory created: %s" %path)

# simulation preparation --------------------------------------------------------------------------------------------------------------------

# infection cost
def CI_cost(a,I):
	h = 0.5*(5+jnp.sqrt(21)) # slope 5
	k = 0.001288575763893496
	return a*(1/h*I+h*(I-0.001)*(1-1/(1+jnp.exp(1000*(I-0.001))))+k)


# mitigation cost
def CM_div(m):
    if whichcost  == "div":
        return	m/(1-m) # divergent cost 
    if whichcost == "divlog":
        return -jnp.log(1-m)
    if whichcost == "divsqrt":
        return (2/jnp.sqrt(1-m)-2)


# simulation
for t_mitchange in np.arange(0,501,5): # times where we change mitigation policy (1 simulation for each time)
    t_mitchange = int(t_mitchange) # convert to int
    
    # preparation stuff for both parts-----------------------------------------------------------------------------------------------------------

    # part 1: before change
    t_beta = np.linspace(0, T_tot, T_tot) # time points for beta
    T_beforechange = T_part1 # optimize for full time
    # T_beforechange is the total time for which we optimize in the first part, NOT the time until the change    
    dt_beforechange = T_beforechange/(T_beforechange-1) # time step
    t_out_beforechange = np.linspace(0, T_beforechange, T_beforechange) # time points for output
    t_solve_ODE_beforechange = np.linspace(0, T_beforechange, T_beforechange) # time points for solving ODE
    t_beta_beforechange = np.linspace(0, T_beforechange, T_beforechange) # time points for beta

    # part 2: after change
    T_afterchange = T_tot-t_mitchange # optimize for time after mit change only
    dt_afterchange = T_afterchange/(T_afterchange-1) # time step
    t_out_afterchange = np.linspace(0, T_afterchange, T_afterchange)
    t_solve_ODE_afterchange = np.linspace(0, T_afterchange, T_afterchange)
    t_beta_afterchange = np.linspace(0, T_afterchange, T_afterchange)
    
    path = path0 + whichcost + '/'
    CM_func = CM_div
    CM_name = whichcost
    CI_func = CI_cost
    
    # actual simulations ----------------------------------------------------------------------------------------------------------------------------------------
    
    # part 1: before change ----------------------------------------------------------------------------------------------------------------------------------------
    # optimize only for variant 1

    # simulation function
    def simulation_beforechange(x): 
        frac_reduc = jax.nn.sigmoid(-x)*0.99999999999999999 # transform values to [0,1]
        beta_t = frac_reduc*beta_0_1_t
        beta_t_func = icomo.interpolate_func(ts_in=t_beta,values=beta_t) # beta is now callable function
        y0 = {'S': 1-I01-0, 'I1': I01, 'I2': 0, 'R':0} # initial conditions
        args = {'gamma': gamma, 'nu': nu, 'beta_diff_factor': 0, 'beta_t': beta_t_func} # arguments for ODE
        
        output = icomo.diffeqsolve(args=args, ODE = SIRS.SIRS_twovariants, y0 = y0, ts_out = t_out_beforechange) # solve ODE

        eff_frac_reduc = icomo.interpolate_func(t_beta_beforechange, frac_reduc, 'cubic')(t_out_beforechange) # interpolate frac_reduc to t_out
        return output.ys, eff_frac_reduc
    
    # cost function for optimization
    @jax.jit
    def min_func_beforechange(x):
        output, frac_reduc = simulation_beforechange(x) # carry out simulation
        m = 1-frac_reduc # mitigation
        cost = jnp.sum(CI_func(a,jnp.add(output['I1'],output['I2'])) + CM_func(m))*dt_beforechange 
        return cost

    # initial values for optimization
    x_0_beforechange = np.ones(T_beforechange)*np.random.rand(T_beforechange)
    M0 = 1-jax.nn.sigmoid(-x_0_beforechange[0])*0.99999999999999999

    # use automatic differentiation for cost function
    value_and_grad_min_func_beforechange = jax.jit(jax.value_and_grad(min_func_beforechange))

    # use L-BFGS-B to optimize the cost function
    solver_beforechange = jaxopt.ScipyMinimize(fun=value_and_grad_min_func_beforechange, value_and_grad=True, method="L-BFGS-B", jit=False, maxiter=1000)#tol = 1e-10, 
    
    # run the optimization
    res_beforechange = solver_beforechange.run(x_0_beforechange)

    # results  
    output_beforechange, frac_reduc_beforechange = simulation_beforechange(res_beforechange.params)
    mitigation_beforechange = 1-frac_reduc_beforechange

    #----------------------------------------------------------------------------------------------------------------------------------------
    # before doing the optimization for the second part (after mitigation adjustment), we need to simulate the part until the mitigation change 
    # with the mitigation result from the first part but the actual beta_0(t) (both variants) instead of the one we used for the 
    # first part (only variant 1). The results from that simulation we can then use as starting values for the second part.

    # simulation function 
    def simulation_beforechange_real(x):
        frac_reduc = jax.nn.sigmoid(-x)*0.99999999999999999 # transform values to [0,1]
        beta_t = frac_reduc*beta_0_1_t
        beta_t_func = icomo.interpolate_func(ts_in=t_beta,values=beta_t) # beta is now callable function
        y0 = {'S': 1-I01-I02, 'I1': I01, 'I2': I02, 'R':0} # use normal starting values
        args = {'gamma': gamma, 'nu': nu, 'beta_diff_factor': beta_diff_factor, 'beta_t': beta_t_func} # arguments for ODE
        
        output = icomo.diffeqsolve(args=args, ODE = SIRS.SIRS_twovariants, y0 = y0, ts_out = t_out_beforechange) # solve ODE

        eff_frac_reduc = icomo.interpolate_func(t_beta_beforechange, frac_reduc, 'cubic')(t_out_beforechange) # interpolate frac_reduc to t_out
        return output.ys, eff_frac_reduc
    
    # results
    output_beforechange_real, frac_reduc_beforechange_real = simulation_beforechange_real(res_beforechange.params) # use mitigation output from sim before change as starting values
    realS = output_beforechange_real['S']
    realI1 = output_beforechange_real['I1']
    realI2 = output_beforechange_real['I2']
    realR = output_beforechange_real['R']

    
    # part 2: after change ----------------------------------------------------------------------------------------------------------------------------------------
    # now optimize for the time after the mitigation adjustment. Bot variants are taken into account.

    # simulation function
    def simulation_afterchange(x):
        frac_reduc = jax.nn.sigmoid(-x)*0.99999999999999999 # transform values to [0,1]
        beta_t = frac_reduc*beta_0_1_t[t_mitchange:]
        beta_t_func = icomo.interpolate_func(ts_in=t_beta_afterchange,values=beta_t) # beta is now callable function
        # use values from first simulation as initial values
        y0 = {'S': realS[t_mitchange-1], 'I1': realI1[t_mitchange-1], 'I2': realI2[t_mitchange-1], 'R': realR[t_mitchange-1]}
        args = {'gamma': gamma, 'nu': nu, 'beta_diff_factor': beta_diff_factor, 'beta_t': beta_t_func} # arguments for ODE
        
        output = icomo.diffeqsolve(args=args, ODE = SIRS.SIRS_twovariants, y0 = y0, ts_out = t_out_afterchange) # solve ODE

        eff_frac_reduc = icomo.interpolate_func(t_beta_afterchange, frac_reduc, 'cubic')(t_out_afterchange) # interpolate frac_reduc to t_out
        return output.ys, eff_frac_reduc
    
    # cost function for optimization
    @jax.jit
    def min_func_afterchange(x):
        output, frac_reduc = simulation_afterchange(x) # carry out simulation
        m = 1-frac_reduc # mitigation
        cost = jnp.sum(CI_func(a,jnp.add(output['I1'],output['I2'])) + CM_func(m))*dt_afterchange
        return cost
    
    # initial values for optimization
    x_0_afterchange = np.ones(len(t_beta_afterchange))*np.random.rand(len(t_beta_afterchange))

    # use automatic differentiation for cost function
    value_and_grad_min_func_afterchange = jax.jit(jax.value_and_grad(min_func_afterchange))

    # use L-BFGS-B to optimize the cost function
    solver_afterchange = jaxopt.ScipyMinimize(fun=value_and_grad_min_func_afterchange, value_and_grad=True, method="L-BFGS-B", jit=False, maxiter=1000)

    # run the optimization
    res_afterchange = solver_afterchange.run(x_0_afterchange)

    # results
    output_afterchange, frac_reduc_afterchange = simulation_afterchange(res_afterchange.params)
    mitigation_afterchange = 1-frac_reduc_afterchange

    # write simulation info file
    f = open('%stmitchange%g_beta1_%g_beta2_%g_multcomp_info.dat' %(path,t_mitchange,beta_0_1,beta_0_2), 'w')	
    f.write("beta_0_1 %f \n beta_0_2 %f \n t_mitchange %f \n gamma %f \n nu %f \n a %f \n I01 %f \n I02 %f \n M0 %f \n T_tot %f \n CM_name %s \n " %(beta_0_1,beta_0_2,t_mitchange,gamma,nu,a,I01,I02,M0,T_tot,CM_name))

    # resulting I and M
    I1 = np.append(realI1[:t_mitchange], output_afterchange['I1']) # use the real values before the change
    I2 = np.append(realI2[:t_mitchange], output_afterchange['I2']) # use the real values before the change
    M = np.append(mitigation_beforechange[:t_mitchange], mitigation_afterchange) # use the mit from the first simulation before the change
    t = np.linspace(0, T_tot, T_tot)
    
    # save data
    with open('%stmitchange%g_beta1_%g_beta2_%g_multcomp_data.npy' %(path,t_mitchange,beta_0_1,beta_0_2), 'wb') as f:
        np.save(f, t)
        np.save(f, np.append(realS[:t_mitchange], output_afterchange['S']))
        np.save(f, I1)
        np.save(f, I2)
        np.save(f, M)
    
    

        