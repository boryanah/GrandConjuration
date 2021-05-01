# How to run chains
`addqueue -c cobaya_0p005_dx1_hsc -O -n 2x12 -q cmb -m 1 /users/damonge/.local/bin/cobaya-run params_marg0p005_diagx1_hsc.yml`

# How to update the code
`python3 setup.py install --user`

# How to run using python
`python3 run.py`

# How to install cobaya
`pip3 install cobaya --user`

Gui interface:

`python3 -m pip install pyqt5 pyside2 --user --upgrade`

- Likelihoods:
`cobaya-install cosmo -m /users/boryanah/repos/GrandConjuration/cobaya_packages`

# How to install other requirements
`pip3 install chaospy --user`
`python3 -m pip install -v git+https://github.com/sfschen/velocileptors`
`python3 -m pip install -v git+https://github.com/kokron/anzu`

# Questions
Do we want `norm` in the yaml file for the bias params?
Why is `anzu` so slow?
Why did `module load mpi` appear in a `.bashrc` file locally?
Why are the Planck likelihoods not all working?
Why do I get this error?
[model] *ERROR* Requirement Cl of planck_2018_lowl.TT is not provided by any component