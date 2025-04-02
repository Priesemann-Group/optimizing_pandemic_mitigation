# Description: This script carries out simulations for the SIRS model with delayed mitigation onset

# import libraries
import os
os.environ["JAX_LOG_COMPILES"] = "0"
import numpy as np
import jax.numpy as jnp
import jax
import jaxopt
import library_costmin.SIRS as SIRS
import icomo # https://icomo.readthedocs.io/en/stable/

# define parameters
whichcost = "div" # which cost function to use, can be 'div', 'divlog', 'divsqrt'
years = 1 # similated time in years
T = int(years*365)+90 # 1 year + 90 days for decrease of beta0
a = 10 # disease severity 
beta_0 = 0.2 # basic reprodition number
gamma = 0.1 # recovery rate
nu = 0.01 # waning rate for immunity after infection
I0 = 1e-4 # starting number of infected individuals



# adding an additional 90 days to the simulation time for the decrease of beta_0 (to soften boundary effects)
beta_0_t = np.concatenate([np.ones(years*365)*beta_0,[beta_0 - x*beta_0/89 for x in range(90)]])

# path for saving files
date = 'costofwaiting_timedep_%gy_a%g' %(years,a)
path0 = './data/costofwaiting_%s/'%whichcost + date + '/'
for costpath in [whichcost]: # checking if path for saving exists, if not create it
    path = path0 + costpath + '/'
    if not os.path.exists(path):
        print("path for saving files does not exist, creating new directory")
        os.makedirs(path)
        print("new directory created: %s" %path)




# simulation preparation -------------------------------------------------------------------------------------------------------------
len_sim = T # simulation length in days
num_points = len_sim # number of points for simulation
dt = len_sim/(num_points-1) # time step
t_out = np.linspace(0, len_sim, num_points) # time points for output
t_beta = np.linspace(0, len_sim, num_points) # time points for time-dependent input (due to time-dependent mitighation in this case)

# infection cost
slope = 5
def CI_cost(a,I):
	h = 0.5*(5+jnp.sqrt(21)) # slope 5
	k = 0.001288575763893496
	return a * (1/h*I+h*(I-0.001)*(1-1/(1+jnp.exp(1000*(I-0.001))))+k)


# mitigation cost
@jax.jit
def CM_div(m):
    if whichcost  == "div":
        return	m/(1-m) # divergent cost 
    if whichcost == "divlog":
        return -jnp.log(1-m) # divergent logarithmic cost
    if whichcost == "divsqrt":
        return (2/jnp.sqrt(1-m)-2) # divergent square cost

# which cost function to use
CM_func = CM_div
CM_name = whichcost
CI_func = CI_cost
path = path0 + whichcost + '/'


# carry out simulations ------------------------------------------------------------------------------------------------------------
for delta in np.arange(0,201,1): # loop over different delays (needed for plotting)

	# delay
	delay_restrictions = int(min(jnp.argwhere(t_beta > delta)[0]))

	# simulation function
	def simulation(x):
		frac_reduc = jax.nn.sigmoid(-x)*0.99999999999999999 # transform values to [0,1]
		frac_reduc = frac_reduc.at[:(delay_restrictions+1)].set(1.) # no mitigation before delay
		beta_t = frac_reduc*beta_0_t # time-dependent beta
		beta_t_func = icomo.interpolate_func(ts_in=t_beta,values=beta_t) # beta is now callable function
		y0 = {'S': 1-I0, 'I': I0, 'R':0} # initial conditions
		args = {'gamma': gamma, 'nu': nu, 'beta_t': beta_t_func} # arguments for ODE
		
		output = icomo.diffeqsolve(args=args, ODE = SIRS.SIRS, y0 = y0, ts_out = t_out) # solve ODE

		eff_frac_reduc = icomo.interpolate_func(t_beta, frac_reduc, 'cubic')(t_out) # interpolate frac_reduc to t_out
		return output.ys, eff_frac_reduc # return output and effective mitigation
	
	# cost function for optimization
	@jax.jit
	def min_func(x):
		output, frac_reduc = simulation(x) # carry out simulation
		m = 1-frac_reduc # mitigation = 1-frac_reduc
		cost = jnp.sum(CI_func(a,output['I']) + CM_func(m))*dt # calculate cost
		return cost
	value_and_grad_min_func = jax.jit(jax.value_and_grad(min_func)) # automatic differentiation for cost function

	x_0 = np.zeros(len(t_beta)) # starting values for optimization
	M0 = 1-jax.nn.sigmoid(-x_0[0])*0.99999999999999999 # initial mitigation

	# use L-BFGS-B algorithm for optimization
	solver = jaxopt.ScipyMinimize(fun=value_and_grad_min_func, value_and_grad=True, method="L-BFGS-B", jit=False, maxiter=500)
	res = solver.run(x_0) # carry out optimization
		
	output, frac_reduc = simulation(res.params) # carry out simulation with optimal mitigation (which is between -inf and inf)
	mitigation = 1-frac_reduc # mitigation = 1-frac_reduc (between 0 and 1)
	
	# save information about simulation
	f = open('%sCIslope%s_cost%s_delta%a_info.dat' %(path,slope,CM_name,delta), 'w')	
	f.write("beta_0 %f \n gamma %f \n nu %f \n a %f \n I0 %f \n M0 %f \n delta %f \n len_sim %f \n num_points %f \n CM_name %s \n res.state.fun_val %f \n " %(beta_0,gamma,nu,a,I0,M0,delta,len_sim,num_points,CM_name,res.state.fun_val))

	# save output
	with open('%sCIslope%s_cost%s_delta%s_data.npy' %(path,slope,CM_name,delta), 'wb') as f:
		np.save(f, t_out)
		np.save(f, output['S'])
		np.save(f, output['I'])
		np.save(f, mitigation)

# no mitigation (for comparison plot) ---------------------------------------------------------------------------------
# simulation without mitigation (Mitigation = 0)
def simulation_nomit():
	y0 = {'S': 1-I0, 'I': I0, 'R':0}
	beta_t_func = icomo.interpolate_func(ts_in=t_beta,values=beta_0_t)
	args = {'gamma': gamma, 'nu': nu, "beta_t": beta_t_func}
	output = icomo.diffeqsolve(args=args, ODE = SIRS.SIRS, y0 = y0, ts_out = t_out)
	return output.ys

output_nomit = simulation_nomit() # carry out simulation without mitigation
# save information about simulation without mitigation
with open('%sCIslope%s_cost%s_nomit_data.npy' %(path,slope,whichcost), 'wb') as f:
	np.save(f, t_out)
	np.save(f, output_nomit['S'])
	np.save(f, output_nomit['I'])