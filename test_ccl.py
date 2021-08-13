import numpy as np
from galshcl_like import CLCCL, CCL, GalshClLike

from cobaya.model import get_model
from cobaya.likelihood import Likelihood
import pyccl as ccl

import yaml

class Tester(Likelihood):
    params = {'b_hydro': {"prior": {"min": 0, "max": 1}}}

    def get_requirements(self):
        return {'CCL': {"methods": {'test_method': self.test_method},
                        "kmax": 10,
                        #"nonlinear": True}}; SOLikeT
                        "external_nonlin_pk": True}}# GalSh

    def test_method(self, cosmo):
        z_n = np.linspace(0., 1., 200)
        n = np.ones(z_n.shape)
        tracer1 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
        tracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
        ell = np.logspace(np.log10(3), 3)
        cls = ccl.cls.angular_cl(cosmo, tracer1, tracer2, ell)
        return cls

    def logp(self, **pars):
        results = self.provider.get_CCL()
        cls = results['test_method']
        np.testing.assert_almost_equal(cls[0], 1.3478e-08, decimal=8)
        return pars['b_hydro']


cosmo_params = {
    "Omega_c": 0.25,
    "Omega_b": 0.05,
    "h": 0.67,
    "n_s": 0.96
}

info = {"params": {"omch2": cosmo_params['Omega_c'] * cosmo_params['h'] ** 2.,
                   "ombh2": cosmo_params['Omega_b'] * cosmo_params['h'] ** 2.,
                   "H0": cosmo_params['h'] * 100,
                   "ns": cosmo_params['n_s'],
                   "As": 2.2e-9,
                   "tau": 0.06,
                   "mnu": 0.06},
        "likelihood": {"Tester": Tester}, # TEST
        "theory": {
            "camb": None,
            #"ccl": {"external": CLCCL, "nonlinear": False} # used to be CCL; SOLikeT
            "ccl": {"external": CLCCL, "external_nonlin_pk": False, "transfer_function": "boltzmann_camb"}, # used to be CCL; GalSh
        },
        "debug": False, "stop_at_error": True}


#fake_likelihood = True
fake_likelihood = False

# load the yaml
config_fn = 'test_config/galsh_heft_frankenstein.yml'
with open(config_fn, "r") as fin:
    info_yaml = yaml.load(fin, Loader=yaml.FullLoader)


#info['likelihood'] = info_yaml['likelihood']


info_comb = {}
for key in info_yaml.keys():
    info_comb[key] = info_yaml[key]

for key in info.keys():
    if key == 'params': continue
    info_comb[key] = info[key]

As = 2.081691e-09
#params = {'ns': 0.9649, 'logA':3.05, 'ombh2': 0.022, 'omch2': 0.12, 'tau': 0.05, 'H0': 70., 'mnu': 0.06}
params = {'ns': 0.9649, 'logA': np.log(As*1.e10), 'ombh2': 0.02236923, 'omch2': 0.1197866, 'tau': 0.055, 'H0': 6.743436e+01, 'mnu': 0.06}
for k, v in info_yaml['params'].items():
    if "DES" in k or 'IA' in k:    
        params[k] = v['ref']['loc']
for key in params.keys():
    # HEFT
    '''
    if 'b0' in key:
        params[key] = 1.    
    elif 'b1' in key:
        if 'gc0' in key:
            params[key] = 0.41
        if 'gc1' in key:
            params[key] = 0.6
        if 'gc2' in key:
            params[key] = 0.6
        if 'gc3' in key:
            params[key] = 0.91
        if 'gc4' in key:
            params[key] = 0.96
    '''
    # Linear
    if 'b0' in key:
        if 'gc0' in key:
            params[key] = 1.41
        if 'gc1' in key:
            params[key] = 1.6
        if 'gc2' in key:
            params[key] = 1.6
        if 'gc3' in key:
            params[key] = 1.91
        if 'gc4' in key:
            params[key] = 1.96
    elif 'bs' in key or 'bn' in key or 'b2' in key or 'dz' in key or 'wl' in key:
        params[key] = 0.
        
print(params)

if fake_likelihood:
    model = get_model(info)
    loglikes, derived = model.loglikes({'b_hydro': 0.3})
else:
    model = get_model(info_yaml)
    loglikes, derived = model.loglikes(params)

print('loglikes =', loglikes)
print("logLkl_MP = ", -289.823)
print('OK')
# matches v well for Linear and not HEFT with parameters as above
