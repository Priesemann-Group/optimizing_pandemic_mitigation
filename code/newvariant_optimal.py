# Description: This script is used to simulate the SIRS model with two variants and find the optimal mitigation strategy. 
# This file is only for the optimal scenario (new variant is known from the start).

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
a = 10  # disease severity
years = 9 # years of simulation
T = years*365+90 # 1 year + 90 days for decrease of beta0
beta_0_1 = 0.2 # beta for variant 1
beta_diff_factor = 1.5 # factor for beta_2
beta_0_2 = beta_0_1*beta_diff_factor # beta for variant 2
beta_0_1_t = np.concatenate([np.ones(years*365)*beta_0_1,[beta_0_1 - x*beta_0_1/89 for x in range(90)]]) # beta_0_1 for the first year and then decrease to 0 over 90 days
gamma = 0.1 # recovery rate
nu = 1/100 # waning rate
# starting values for variant 1 and 2, given are the best starting values for different cost functions etc so that we dont have long times until equilibrium is reached
#I01 = 0.00083 # for beta=0.2 and a=10 div
I01 = 0.00034955773027785253 # for beta=0.2 and a=10, divlog and divsqrt
#I01 = 0.04554786472600939 # for beta=0.2 and a=2
#I01 = 0.0001127868959310304 # for beta=0.2 and a=50
#I01 = 0.0454556834290207 # for beta=0.2 and a=1/0.3
#I01 = 0.001808440551412749 # for beta=0.2 and a=5
I02 = 0.0000001 # starting value for variant 2

# where to save the data
date = 'newvariant_known_%gyr_a%g' %(years,a)
path0 = './data/newvariant_multcomp_%s/'%whichcost + date + '/' 
for costpath in [whichcost]: # check if the path exists and create it if not
    path = path0 + costpath + '/'
    if not os.path.exists(path):
        print("path for saving files does not exist, creating new directory")
        os.makedirs(path)
        print("new directory created: %s" %path)



# simulation preparation --------------------------------------------------------------------------------------------------------------------

len_sim = T
num_points = T
dt = len_sim/(num_points-1)
t_out = np.linspace(0, len_sim, num_points)
t_beta = np.linspace(0, len_sim, num_points)


# infection cost
def CI_cost(a,I):
	h = 0.5*(5+jnp.sqrt(21)) # slope 5
	k = 0.001288575763893496
	return a*(1/h*I+h*(I-0.001)*(1-1/(1+jnp.exp(1000*(I-0.001))))+k)


# mitigation cost
@jax.jit
def CM_div(m):
    if whichcost  == "div":
        return	m/(1-m) # divergent cost 
    if whichcost == "divlog":
        return -jnp.log(1-m)
    if whichcost == "divsqrt":
        return (2/jnp.sqrt(1-m)-2)

# path and cost name (don't change)
path = path0 + whichcost + '/'
CM_func = CM_div
CM_name = whichcost
CI_func = CI_cost

# simulation function
def simulation(x):
    frac_reduc = jax.nn.sigmoid(-x)*0.99999999999999999 # sigmoid function to get values between 0 and 1
    beta_t = frac_reduc*beta_0_1_t
    beta_t_func = icomo.interpolate_func(ts_in=t_beta,values=beta_t) # beta is now callable function
    y0 = {'S': 1-I01-I02, 'I1': I01, 'I2': I02, 'R':0} # initial conditions
    args = {'gamma': gamma, 'nu': nu, 'beta_diff_factor': beta_diff_factor, 'beta_t': beta_t_func} # arguments for ODE
    
    output = icomo.diffeqsolve(args=args, ODE = SIRS.SIRS_twovariants, y0 = y0, ts_out = t_out) # solve ODE

    eff_frac_reduc = icomo.interpolate_func(t_beta, frac_reduc, 'cubic')(t_out) # interpolate frac_reduc to t_out
    return output.ys, eff_frac_reduc

# simultaion with no mitigation
def simulation_nomit():
    frac_reduc = np.ones(len(t_beta)) # no mitigation
    beta_t = beta_0_1_t
    beta_t_func = icomo.interpolate_func(ts_in=t_beta,values=beta_t) # beta is now callable function
    y0 = {'S': 1-I01-I02, 'I1': I01, 'I2': I02, 'R':0} # initial conditions
    args = {'gamma': gamma, 'nu': nu, 'beta_diff_factor': beta_diff_factor, 'beta_t': beta_t_func} # arguments for ODE
    
    output = icomo.diffeqsolve(args=args, ODE = SIRS.SIRS_twovariants, y0 = y0, ts_out = t_out) # solve ODE

    eff_frac_reduc = icomo.interpolate_func(t_beta, frac_reduc, 'cubic')(t_out) # interpolate frac_reduc to t_out
    return output.ys, eff_frac_reduc

# cost function for optimization
@jax.jit
def min_func(x):
    output, frac_reduc = simulation(x) # carry out simulation
    m = 1-frac_reduc # mitigation
    cost = jnp.sum(CI_func(a,jnp.add(output['I1'],output['I2'])) + CM_func(m))*dt # I = I1 + I2
    return cost

# starting values for optimization
x_0 = np.zeros(len(t_beta))
M0 = 1-jax.nn.sigmoid(-x_0[0])*0.99999999999999999

# automatic differentiation for cost function
value_and_grad_min_func = jax.jit(jax.value_and_grad(min_func))

# use L-BFGS-B to optimize the cost function
solver = jaxopt.ScipyMinimize(fun=value_and_grad_min_func, value_and_grad=True, method="L-BFGS-B", jit=False, maxiter=1000)

# simulation ----------------------------------------------------------------------------------------------------------------------------------
res = solver.run(x_0) # carry out optimization

# results
output, frac_reduc = simulation(res.params)
mitigation = 1-frac_reduc
output_nomit, frac_reduc_nomit = simulation_nomit()
mitigation_nomit = 1-frac_reduc_nomit

# save information about simulation
f = open('%sa%g_beta1_%g_beta2_%g_multcomp_info.dat' %(path,a,beta_0_1,beta_0_2), 'w')	
f.write("beta_0_1 %f \n beta_0_2 %f \n gamma %f \n nu %f \n a %f \n I01 %f \n I02 %f \n M0 %f \n len_sim %f \n num_points %f \n CM_name %s \n res.state.fun_val %f \n " %(beta_0_1,beta_0_2,gamma,nu,a,I01,I02,M0,len_sim,num_points,CM_name,res.state.fun_val))

# save results
I1 = output['I1']
I2 = output['I2']
M = mitigation
t = t_out
with open('%sa%g_beta1_%g_beta2_%g_multcomp_data.npy' %(path,a,beta_0_1,beta_0_2), 'wb') as f:
    np.save(f, t)
    np.save(f, output['S'])
    np.save(f, I1)
    np.save(f, I2)
    np.save(f, M)



    