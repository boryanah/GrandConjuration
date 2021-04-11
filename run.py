#from shcl_like.ccl import CCL
from galshcl_like.ccl import CCL
from cobaya.model import get_model
from cobaya.run import run
import yaml
import os



#with open('test_data/buba.yml', "r") as fin:
with open('test_data/galsh.yml', "r") as fin:
    info = yaml.load(fin, Loader=yaml.FullLoader)

p0 = {}
for p in info['params']:
     if isinstance(info['params'][p], dict):
         if 'ref' in info['params'][p]:
             p0[p] = info['params'][p]['ref']['loc']

os.system('mkdir -p ' + info['output'])

for key in p0.keys():
    if 'b0' not in key:
        p0[key] = 0.
p0['Omega_c'] = 2.634181e-01
p0['Omega_b'] = 4.919133e-02
p0['h'] = 6.743436e-01
p0['n_s'] = 0.9649
p0['A_sE9'] = 2.081691e+00

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

print(p0)

model = get_model(info)
loglikes, derived = model.loglikes(p0)
print(derived)
#print(-2 * loglikes)
print("logLkl = ", -loglikes)
print("logLkl_MP = ", 289.823)
