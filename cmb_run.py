#from shcl_like.ccl import CCL
from galshcl_like import CLCCL, CCL
#from galshcl_like.clccl import CLCCL
from cobaya.model import get_model
from cobaya.run import run
import yaml
import os

# Read in the yaml file
#config_fn = 'test_data/galsh.yml'
config_fn = 'test_data/planck.yml'
#config_fn = 'test_data/galsh_heft.yml'
with open(config_fn, "r") as fin:
    info = yaml.load(fin, Loader=yaml.FullLoader)

# Get the mean proposed in the yaml file for each parameter
p0 = {}
for p in info['params']:
     if isinstance(info['params'][p], dict):
         if 'ref' in info['params'][p]:
             p0[p] = info['params'][p]['ref']['loc']
#os.system('mkdir -p ' + info['output'])
print(p0)

# Compute the likelihood
model = get_model(info)
loglikes, derived = model.loglikes(p0)
print("sigma8 = ", derived)
print("chi2 = ", -2 * loglikes)
print("logLkl = ", -loglikes)
print("logLkl_MP = ", 289.823)
