from shcl_like.ccl import CCL
from cobaya.model import get_model
from cobaya.run import run
import yaml
import os


with open('test_data/buba.yml', "r") as fin:
    info = yaml.load(fin, Loader=yaml.FullLoader)

p0 = {}
for p in info['params']:
     if isinstance(info['params'][p], dict):
         if 'ref' in info['params'][p]:
             p0[p] = info['params'][p]['ref']['loc']

os.system('mkdir -p ' + info['output'])

print(p0)
model = get_model(info)
loglikes, derived = model.loglikes(p0)
print(derived)
print(-2 * loglikes)