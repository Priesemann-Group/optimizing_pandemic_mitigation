# Description: This script carries out simulations for the SIRS model with vaccination and multiple compartments.
# The model is defined in library_costmin/SIRS.py.
# Simulations are automatically run for different vaccine efficiencies between 0 and 1.

# import libraries
import os
os.environ["JAX_LOG_COMPILES"] = "0"
import numpy as np
import jax.numpy as jnp
import jax
import jaxopt
import library_costmin.SIRS as SIRS
import matplotlib.pyplot as plt
import icomo # https://icomo.readthedocs.io/en/stable/

# define parameters
whichcost = 'div' # which cost function to use, can be 'div', 'divlog', 'divsqrt'
a = 10 # disease severity 
years = 3 # similated time in years
T = years*365 # number of days
date = 'vacc_%gyr_a%g' %(years,a) # name for saving files
path0 = './data/vacc_multcomp_%s_0.4max_3years/'%whichcost + date + '/'  # path for saving files
for costpath in [whichcost]: # create path for saving files if it does not exist
    path = path0 + costpath + '/'
    if not os.path.exists(path):
        print("path for saving files does not exist, creating new directory")
        os.makedirs(path)
        print("new directory created: %s" %path)

beta_0_1 = 0.2 # basic reprodiction number
gamma = 0.1 # recovery rate
nu = 1/100 # waning rate for immunity after infection
nuv = 1/150 # waning rate for immunity after vaccination
I0 = 0.00128500683512649 # starting number of infected individuals

# simulation preparation
len_sim = T # length of simulation in days
num_points = T # number of points for simulation
dt = len_sim/(num_points-1) # time step
t_out = np.linspace(0, len_sim, num_points) # time points for output
t_solve_ODE = np.linspace(0, len_sim, num_points) # time points for solving ODE
t_beta = np.linspace(0, len_sim, num_points) # time points for time-dependent input (due to time-dependent mitighation in this case)
delay_restrictions = int(min(jnp.argwhere(t_beta >= 0)[0])) # no delay
cost_weighting = 1
benchmark=True

# define cost functions
# infection cost CI
def CI_cost(a,I):
	h = 0.5*(5+jnp.sqrt(21)) 
	k = 0.001288575763893496
	return a*(1/h*I+h*(I-0.001)*(1-1/(1+jnp.exp(1000*(I-0.001))))+k)


# different mitigation cost functions CM
@jax.jit
def CM_div(m):
    if whichcost  == "div":
        return	m/(1-m) # standard
    if whichcost == "divlog":
        return -jnp.log(1-m)
    if whichcost == "divsqrt":
        return (2/jnp.sqrt(1-m)-2)


# carry out simulations
for vacc_eff in np.arange(0,1.01,0.05): # loop over different vaccination efficacies (needed for plotting)
    
    # define parameters
    path = path0 + whichcost + '/'
    CM_func = CM_div
    CM_name = whichcost
    CI_func = CI_cost
    
    # define simulation for given vaccination efficacy
    def simulation(x): # x is the mitigation function, can be between -inf and inf
        frac_reduc = jax.nn.sigmoid(-x)*0.99999999999999999 # transform values to [0,1]
        beta_t = frac_reduc*beta_0_1 # effective transmission rate beta_eff(t) = beta_0*frac_reduc(t), frac_reduc(t) = 1-M(t)
        beta_t_func = icomo.interpolate_func(ts_in=t_beta,values=beta_t)
        y0 = {'S': 1-I0, 'Sv': 0, 'I': I0, 'Iv': 0, 'R':0, 'Rv':0} # initial conditions for SIRS model
        args = {'gamma': gamma, 'nu': nu, "eta": vacc_eff, 'nuv': nuv, 'beta_t': beta_t_func} # constant arguments for SIRS model
        
        # carry out simulation
        #output = jax.jit(SIRS_vacc_integrator)(y0=y0, arg_t=beta_t, constant_args=const_args)
        output = icomo.diffeqsolve(args=args, ODE = SIRS.SIRS_vacc_multcomp, y0 = y0, ts_out = t_out)

        eff_frac_reduc = icomo.interpolate_func(t_beta, frac_reduc, 'cubic')(t_out) # interpolate mitigation function to output time points
        return output.ys, eff_frac_reduc # return output (SIRS(t)) and frac_reduc (\in[0,1])
        
    @jax.jit
    def min_func(x): # define cost function to minimize, x is the mitigation function
        output, frac_reduc = simulation(x) # carry out simulation
        m = 1-frac_reduc # mitigation
        cost = jnp.sum(CI_func(a,jnp.add(output['I'],output['Iv'])) + CM_func(m))*dt # calculate cost
        return cost # return cost

    # starting values for optimization
    x_0 = np.zeros(len(t_beta)) # starting values for mitigation function
    M0 = 1-jax.nn.sigmoid(-x_0[0])*0.99999999999999999 # starting values for mitigation function

    # preparation for optimization
    # calculate value and gradient of cost function (with automatic differentiation from jax)
    value_and_grad_min_func = jax.jit(jax.value_and_grad(min_func)) 
    # use scipy optimization method L-BFGS-B
    solver = jaxopt.ScipyMinimize(fun=value_and_grad_min_func, value_and_grad=True, method="L-BFGS-B", jit=False, maxiter=1000)#tol = 1e-10, 
    
    # carry out optimization
    res = solver.run(x_0)       
    output, frac_reduc = simulation(res.params) # carry out simulation with optimized mitigation function
    mitigation = 1-frac_reduc # optimal mitigation 

    # saving results
    # save parameters to info file
    f = open('%sa%g_beta1_%g_vacceff%g_info.dat' %(path,a,beta_0_1,vacc_eff), 'w')	
    f.write("beta_0_1 %f \n gamma %f \n nu %f \n a %f \n I0 %f \n M0 %f \n len_sim %f \n num_points %f \n CM_name %s \n res.state.fun_val %f \n " %(beta_0_1,gamma,nu,a,I0,M0,len_sim,num_points,CM_name,res.state.fun_val))
    I = output['I']
    M = mitigation
    t = t_out
    # save results to npy file
    with open('%sa%g_beta1_%g_vacceff%g_nuv%g_data.npy' %(path,a,beta_0_1,vacc_eff,nuv), 'wb') as f:
        np.save(f, t)
        np.save(f, output['S'])
        np.save(f, output['Sv'])
        np.save(f, I)
        np.save(f, output['Iv'])
        np.save(f, M)
 

        