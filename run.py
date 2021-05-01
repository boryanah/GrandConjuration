#from shcl_like.ccl import CCL
from galshcl_like import CLCCL, CCL
#from galshcl_like.clccl import CLCCL
from cobaya.model import get_model
from cobaya.run import run
import yaml
import os

# Read in the yaml file
#config_fn = 'test_data/galsh.yml'
#config_fn = 'test_data/buba.yml'
config_fn = 'test_data/galsh_heft.yml'
with open(config_fn, "r") as fin:
    info = yaml.load(fin, Loader=yaml.FullLoader)

# Get the mean proposed in the yaml file for each parameter
p0 = {}
for p in info['params']:
     if isinstance(info['params'][p], dict):
         if 'ref' in info['params'][p]:
             p0[p] = info['params'][p]['ref']['loc']
os.system('mkdir -p ' + info['output'])

# Modify the parameter values so you can compare to MontePython below
for key in p0.keys():
    if 'b0' in key:
        print(key, p0[key])
        p0[key] = 1.
    elif 'b1' in key:
        if 'gc0' in key:
            p0[key] = 0.41
        if 'gc1' in key:
            p0[key] = 0.6
        if 'gc2' in key:
            p0[key] = 0.6
        if 'gc3' in key:
            p0[key] = 0.91
        if 'gc4' in key:
            p0[key] = 0.96
    else:
        p0[key] = 0.
        
p0['Omega_c'] = 2.634181e-01
p0['Omega_b'] = 4.919133e-02
p0['h'] = 6.743436e-01
p0['n_s'] = 0.9649
p0['A_sE9'] = 2.081691e+00

# MontePython output
##  logLkl       omega_b         omega_cdm       sigma8_cb       n_s
#1  289.823      2.236923e-02    1.197866e-01    8.111000e-01    9.649000e-01
#gc_b0_0             gc_b0_1         gc_b0_2         gc_b0_3         gc_b0_4
#1.410000e+00    1.600000e+00   1.600000e+00    1.910000e+00    1.960000e+00  
#gc_dz_0             gc_dz_1         gc_dz_2         gc_dz_3         gc_dz_4
#0.000000e+00    0.000000e+00    0.000000e+00    0.000000e+00    0.000000e+00
#wl_ia_A             wl_ia_eta       wl_m_0          wl_m_1          wl_m_2
#0.000000e+00    0.000000e+00    0.000000e+00    0.000000e+00    0.000000e+00
#wl_m_3              wl_dz_0         wl_dz_1         wl_dz_2         wl_dz_3
#0.000000e+00    0.000000e+00    0.000000e+00    0.000000e+00    0.000000e+00
#A_s                 Omega_m         S_8             Omega_c         Omega_b         H0
#2.081691e+00    3.126094e-01    8.354837e-01    2.634181e-01    4.919133e-02    6.743436e+01
print("params_dict = ", p0)

# Compute the likelihood
model = get_model(info)
loglikes, derived = model.loglikes(p0)
print("sigma8 = ", derived)
print("chi2 = ", -2 * loglikes)
print("logLkl = ", -loglikes)
print("logLkl_MP = ", 289.823)
