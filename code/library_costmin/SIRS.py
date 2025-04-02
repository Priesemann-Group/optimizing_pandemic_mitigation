# all versions of SIRS functions:
# SIRS 			normal SIRS
# SIRS_seas		SIRS with seasonality
import jax.numpy as jnp


def SIRS(t, y, args):
    β = args["beta_t"]
    γ = args["gamma"]
    nu = args["nu"]
    dS = -β(t) * y["I"] * y["S"] + nu * y["R"]
    dI = β(t) * y["I"] * y["S"]- γ * y["I"]
    dR = γ * y["I"] - nu * y["R"]
    dy = {"S": dS, "I": dI, "R": dR}
    return dy
    
def SIR(t, y, args):
    β, const_arg = args
    γ = const_arg["gamma"]
    nu = const_arg["nu"]
    dS = -β(t) * y["I"] * y["S"] 
    dI = β(t) * y["I"] * y["S"]- γ * y["I"]
    dR = γ * y["I"] 
    dy = {"S": dS, "I": dI, "R": dR}
    return dy

def SIRS_seas(t, y, args):
    period = 1 # period of the seasonality in years
    frac_reduc = args["frac_reduc"]
    γ = args["gamma"]
    nu = args["nu"]
    β_0 = args["beta_0"]
    delta_beta = args["delta_beta"]
    phi = args["phi"]
    dS = -(β_0+delta_beta*jnp.sin(2*jnp.pi*t/(365*period)+phi))*frac_reduc(t) * y["I"] * y["S"] + nu * y["R"]
    dI = (β_0+delta_beta*jnp.sin(2*jnp.pi*t/(365*period)+phi))*frac_reduc(t) * y["I"] * y["S"]- γ * y["I"]
    dR = γ * y["I"] - nu * y["R"]
    dy = {"S": dS, "I": dI, "R": dR}
    return dy

def SIRS_twovariants(t, y, args): 
    beta_t = args["beta_t"]
    γ = args["gamma"]
    nu = args["nu"]
    beta_diff_factor = args["beta_diff_factor"]
    dS = -beta_t(t) * y["I1"] * y["S"] - beta_t(t)*beta_diff_factor * y["I2"] * y["S"] + nu * y["R"]
    dI1 = beta_t(t) * y["I1"] * y["S"]- γ * y["I1"]
    dI2 = beta_t(t)*beta_diff_factor * y["I2"] * y["S"]- γ * y["I2"]
    dR = γ * y["I1"] +  γ * y["I2"] - nu * y["R"]
    dy = {"S": dS, "I1": dI1, "I2": dI2, "R": dR}
    return dy

def SIRS_twovariants_influx(t, y, args): 
    beta_t, const_arg = args
    γ = const_arg["gamma"]
    nu = const_arg["nu"]
    t_mid = const_arg["t_mid"]
    maxinflux = const_arg["maxinflux"]
    timeshift = const_arg["timeshift"]
    t = t+timeshift
    influx = maxinflux/(1+1*jnp.exp(-0.1*(t-t_mid)))
    beta_diff_factor = const_arg["beta_diff_factor"]
    #dS = -beta_t(t) * (y["I1"]+influx) * y["S"] -beta_t(t)*beta_diff_factor * (y["I2"]+influx) * y["S"] + nu * y["R"]
    #dI1 = beta_t(t) * (y["I1"]+influx) * y["S"]- γ * y["I1"]
    #dI2 = beta_t(t)*beta_diff_factor * (y["I2"]+influx) * y["S"]- γ * y["I2"]
    dS = -beta_t(t) * y["I1"] * y["S"] -beta_t(t)*beta_diff_factor * y["I2"] * y["S"] + nu * y["R"] - influx
    dI1 = beta_t(t) * y["I1"] * y["S"]- γ * y["I1"] + influx
    dI2 = beta_t(t)*beta_diff_factor * y["I2"] * y["S"]- γ * y["I2"]
    dR = γ * y["I1"] + γ * y["I2"] - nu * y["R"]
    dy = {"S": dS, "I1": dI1, "I2": dI2, "R": dR}
    return dy

def SIRS_vacc(t, y, args):
    β, const_arg = args
    γ = const_arg["gamma"]
    nu = const_arg["nu"]
    eta = const_arg["eta"]
    N_vacc = 0.003/(1+jnp.exp(-0.1*(t-150)))#0.01*((1/(1+jnp.exp(-0.1*(t-100))))*(-0.5*(1/(1+jnp.exp(-0.1*(t-200))))+1))
    dS = -β(t) * y["I"] * y["S"] + nu * y["R"] + nu *y["Sv"] - N_vacc 
    dSv = -β(t)*(1-eta) * y["I"] * y["Sv"] - nu *y["Sv"] + N_vacc 
    dI = β(t) * y["I"] * y["S"] + β(t)*(1-eta) * y["I"] * y["Sv"] - γ * y["I"]
    dR = γ * y["I"] - nu * y["R"]
    dy = {"S": dS, "Sv": dSv, "I": dI, "R": dR}
    return dy

def SIRS_vacc_multcomp(t, y, args):
    β = args["beta_t"]
    γ = args["gamma"]
    nu = args["nu"]
    nuv = args["nuv"]
    eta = args["eta"]
    N_vacc = 0.004/(1+jnp.exp(-0.1*(t-150)))#0.01*((1/(1+jnp.exp(-0.1*(t-100))))*(-0.5*(1/(1+jnp.exp(-0.1*(t-200))))+1))
    dS = -β(t) * y["I"] * y["S"]  -β(t) * (1-eta) * y["Iv"] * y["S"] + nu * y["R"]  + nuv *y["Sv"] - N_vacc 
    dSv = -β(t)*(1-eta) * y["I"] * y["Sv"] -β(t)*(1-eta)**2 * y["Iv"] * y["Sv"] + nu * y["Rv"] - nuv *y["Sv"] + N_vacc 
    dI = β(t) * y["I"] * y["S"] + β(t)*(1-eta) * y["Iv"] * y["S"] - γ * y["I"]
    dIv = β(t)*(1-eta) * y["I"] * y["Sv"] + β(t)*(1-eta)**2 * y["Iv"] * y["Sv"] - γ * y["Iv"]
    dR = γ * y["I"] - nu * y["R"]
    dRv = γ * y["Iv"] - nu * y["Rv"]
    dy = {"S": dS, "Sv": dSv, "I": dI, "Iv": dIv, "R": dR, "Rv": dRv}
    return dy