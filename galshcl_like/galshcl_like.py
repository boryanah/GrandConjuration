import time

import numpy as np
from scipy.interpolate import interp1d

import pyccl as ccl
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from anzu.emu_funcs import LPTEmulator

class GalshClLike(Likelihood):
    # All parameters starting with this will be
    # identified as belonging to this stage.
    input_params_prefix: str = ""
    # Input sacc file
    input_file: str = ""
    # IA model name. Currently all of these are
    # just flags, but we could turn them into
    # homogeneous systematic classes.
    ia_model: str = "IANone"
    # N(z) model name
    nz_model: str = "NzNone"
    # b(z) model name
    bz_model: str = "BzNone"
    # P(k) model name
    pk_model: str = "PkDefault"
    # List of bin names
    bins: list = []
    # List of default settings (currently only scale cuts)
    defaults: dict = {}
    # List of two-point functions that make up the data vector
    twopoints: list = []

    def initialize(self):
        # Read SACC file
        self._read_data()
        # Ell sampling for interpolation
        self._get_ell_sampling()
        # Initialize HEFT model if used
        if self.bz_model == 'HEFT':
            self._initialize_HEFT()

    def _initialize_HEFT(self):
        # Initialize Hybrid EFT emulator
        self.emu = LPTEmulator(use_sigma_8=False)

        # k values over which tempaltes are computed
        self.k = np.logspace(-3, 0, 1000)

        # If we don't pass redshift (scale factor), emu assumes:
        # [3.0, 2.0, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0] but zmax = 2.0!, so 9
        z = np.array([2.0, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0])
        self.a = 1./(1. + z)

        # Available fields for the emulator
        self.b_emu = ['b0', 'b1', 'b2', 'bs']

        # Create a dictionary for the emulator outputs
        self.emu_dict = {}
        counter = 0
        for i in range(len(self.b_emu)):
            for j in range(len(self.b_emu)):
                if j > i: continue
                self.emu_dict[self.b_emu[i] + '_' + self.b_emu[j]] = counter
                self.emu_dict[self.b_emu[j] + '_' + self.b_emu[i]] = counter
                counter += 1
        
    def get_suffix_for_tr(self, tr):
        # Get name of the power spectra in the sacc file
        if ('gc' in tr) or ('cv' in tr):
            return '0'
        elif ('wl' in tr) or ('bin' in tr):
            return 'e'
        else:
            raise ValueError('dtype not found for tracer {}'.format(tr))

        
    def _read_data(self):
        # Reads sacc file
        # Selects relevant data.
        # Applies scale cuts
        # Reads tracer metadata (N(z))
        # Reads covariance
        import sacc
        s = sacc.Sacc.load_fits(self.input_file)
        self.bin_properties = {}
        for b in self.bins:
            if b['name'] not in s.tracers:
                raise LoggedError(self.log, "Unknown tracer %s" % b['name'])
            t = s.tracers[b['name']]
            self.bin_properties[b['name']] = {'z_fid': t.z,
                                              'nz_fid': t.nz}

        indices = []
        for cl in self.twopoints:
            #lmin = cl.get('lmin', self.defaults.get('lmin', 2))
            #lmax = cl.get('lmax', self.defaults.get('lmax', 1E30))
            lmin = np.min([self.defaults[cl['bins'][0]].get('lmin', 2), self.defaults[cl['bins'][1]].get('lmin', 2)])
            lmax = np.min([self.defaults[cl['bins'][0]].get('lmax', 1E30), self.defaults[cl['bins'][1]].get('lmax', 1E30)])
            # Get the suffix for both tracers
            cl_name1 = self.get_suffix_for_tr(cl['bins'][0])
            cl_name2 = self.get_suffix_for_tr(cl['bins'][1])
            ind = s.indices('cl_%s%s' % (cl_name1, cl_name2), (cl['bins'][0], cl['bins'][1]),
                            ell__gt=lmin, ell__lt=lmax)
            indices += list(ind)
        s.keep_indices(np.array(indices))

        indices = []
        self.cl_meta = []
        id_sofar = 0
        self.used_tracers = []
        self.l_min_sample = 1E30
        self.l_max_sample = -1E30
        for cl in self.twopoints:
            # Get the suffix for both tracers
            cl_name1 = self.get_suffix_for_tr(cl['bins'][0])
            cl_name2 = self.get_suffix_for_tr(cl['bins'][1])
            l, c_ell, cov, ind = s.get_ell_cl('cl_%s%s' % (cl_name1, cl_name2),
                                              cl['bins'][0],
                                              cl['bins'][1],
                                              return_cov=True,
                                              return_ind=True)
            if c_ell.size > 0:
                if cl['bins'][0] not in self.used_tracers:
                    self.used_tracers.append(cl['bins'][0])
                if cl['bins'][1] not in self.used_tracers:
                    self.used_tracers.append(cl['bins'][1])

            bpw = s.get_bandpower_windows(ind)
            if np.amin(bpw.values) < self.l_min_sample:
                self.l_min_sample = np.amin(bpw.values)
            if np.amax(bpw.values) > self.l_max_sample:
                self.l_max_sample = np.amax(bpw.values)

            self.cl_meta.append({'bin_1': cl['bins'][0],
                                 'bin_2': cl['bins'][1],
                                 'l_eff': l,
                                 'cl': c_ell,
                                 'cov': cov,
                                 'inds': (id_sofar +
                                          np.arange(c_ell.size,
                                                    dtype=int)),
                                 'l_bpw': bpw.values,
                                 'w_bpw': bpw.weight.T})
            indices += list(ind)
            id_sofar += c_ell.size
        indices = np.array(indices)
        self.data_vec = s.mean[indices]
        self.cov = s.covariance.covmat[indices][:, indices]
        self.inv_cov = np.linalg.inv(self.cov)
        self.ndata = len(self.data_vec)

    def _get_ell_sampling(self, nl_per_decade=30):
        # Selects ell sampling.
        # Ell max/min are set by the bandpower window ells.
        # It currently uses simple log-spacing.
        # nl_per_decade is currently fixed at 30
        if self.l_min_sample == 0:
            l_min_sample_here = 2
        else:
            l_min_sample_here = self.l_min_sample
        nl_sample = int(np.log10(self.l_max_sample / l_min_sample_here) *
                        nl_per_decade)
        l_sample = np.unique(np.geomspace(l_min_sample_here,
                                          self.l_max_sample+1,
                                          nl_sample).astype(int)).astype(float)

        if self.l_min_sample == 0:
            self.l_sample = np.concatenate((np.array([0.]), l_sample))
        else:
            self.l_sample = l_sample

    def _eval_interp_cl(self, cl_in, l_bpw, w_bpw):
        # Interpolates spectrum and evaluates it at bandpower window
        # ell values.
        f = interp1d(self.l_sample, cl_in)
        cl_unbinned = f(l_bpw)
        cl_binned = np.dot(w_bpw, cl_unbinned)
        return cl_binned

    def _get_nz(self, cosmo, name, **pars):
        # Get an N(z) for tracer with name `name`
        z = self.bin_properties[name]['z_fid']
        nz = self.bin_properties[name]['nz_fid']
        if self.nz_model == 'NzShift':
            z = z + pars[self.input_params_prefix + '_' + name + '_dz']
            msk = z >= 0
            z = z[msk]
            nz = nz[msk]
        elif self.nz_model != 'NzNone':
            raise LoggedError(self.log, "Unknown Nz model %s" % self.nz_model)
        return (z, nz)

    def _get_bz(self, cosmo, name, **pars):
        # Get b(z) for tracer with name `name`
        z = self.bin_properties[name]['z_fid']
        bz = np.ones_like(z)
        if self.bz_model == 'Linear':
            b0 = pars[self.input_params_prefix + '_' + name + '_b0']
            bz *= b0
        elif self.bz_model == 'HEFT':
            # Pass just ones for the bias
            pass
        elif self.bz_model != 'BzNone':
            raise LoggedError(self.log, "Unknown Bz model %s" % self.bz_model)
        return (z, bz)

    def _get_ia_bias(self, cosmo, name, **pars):
        # Get an IA amplitude for tracer with name `name`
        if self.ia_model == 'IANone':
            return None
        else:
            z = self.bin_properties[name]['z_fid']
            if self.ia_model == 'IAPerBin':
                A = pars[self.input_params_prefix + '_' + name + '_A_IA']
                A_IA = np.ones_like(z) * A
            elif self.ia_model == 'IADESY1':
                A0 = pars[self.input_params_prefix + '_A_IA']
                eta = pars[self.input_params_prefix + '_eta_IA']
                A_IA = A0 * ((1+z)/1.62)**eta
            else:
                raise LoggedError(self.log, "Unknown IA model %s" %
                                  self.ia_model)
            return (z, A_IA)

    def _get_tracer(self, cosmo, name, **pars):
        # Get CCL tracer for tracer with name `name`
        if 'cv' not in name:
            nz = self._get_nz(cosmo, name, **pars)
        if 'gc' in name:
            bz = self._get_bz(cosmo, name, **pars)
            t = ccl.NumberCountsTracer(cosmo, dndz=nz, bias=bz, has_rsd=False)
        elif 'wl' in name:
            ia = self._get_ia_bias(cosmo, name, **pars)
            t = ccl.WeakLensingTracer(cosmo, nz, ia_bias=ia)
        elif 'cv' in name:
            # B.H. TODO: pass z_source as parameter to the YAML file
            t = ccl.CMBLensingTracer(cosmo, z_source=1100)
        return t

    def _get_pk(self, cosmo):
        # Get P(k) to integrate over
        if self.pk_model == 'PkDefault':
            return None
        elif self.pk_model == 'PkHModel':
            mdef = ccl.halos.MassDef200c()
            hmf = ccl.halos.MassFuncTinker08(cosmo, mass_def=mdef)
            hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef)
            hmc = ccl.halos.HMCalculator(cosmo, hmf, hbf, mdef)
            cM = ccl.halos.ConcentrationDuffy08(mdef=mdef)
            prof = ccl.halos.HaloProfileNFW(cM)
            lk_s = np.log(np.geomspace(1E-4, 1E2, 256))
            a_s = 1./(1+np.linspace(0., 3., 20)[::-1])
            pk2d = ccl.halos.halomod_Pk2D(cosmo, hmc, prof,
                                          lk_arr=lk_s, a_arr=a_s,
                                          normprof1=True)
            return pk2d
        else:
            raise LoggedError("Unknown power spectrum model %s" %
                              self.pk_model)


    def _parse_cosmo(self, cosmo):
        # Initialize cosmology vector
        cosmovec = np.zeros(8)
        cosmovec[0] = cosmo['Omega_b'] * (cosmo['H0'] / 100)**2
        cosmovec[1] = cosmo['Omega_c'] * (cosmo['H0'] / 100)**2
        cosmovec[2] = cosmo['w0']
        cosmovec[3] = cosmo['n_s']
        cosmovec[4] = np.log(cosmo['A_s'] * 1.e10)
        cosmovec[5] = cosmo['H0']
        cosmovec[6] = cosmo['Neff']
        cosmovec[7] = 1. # random scale factor just to initialize
                                    
        # Vector of cosmological parameters
        cosmovec = np.atleast_2d(cosmovec)

        return cosmovec


    def _compute_emu_spec(self, cosmo):
        # 9 redshifts, 10 combinations between bias params, and the rest are the ks
        num_comb = int(len(self.b_emu)*(len(self.b_emu)-1)/2 + len(self.b_emu))
        emu_spec = np.zeros((len(self.a), num_comb, len(self.k)))
        
        # Get the emulator prediction for this cosmology
        cosmovec = self._parse_cosmo(cosmo)
            
        for i in range(len(self.a)):
            cosmovec[-1, -1] = self.a[i]
            emu_spec[i] = self.emu.predict(self.k, cosmovec)

        return emu_spec

    def _get_p_of_k_a(self, emu_spec, cosmo, b_trs, clm):
        
        # Initialize power spectrum Pk_a(as, ks)
        Pk_a = np.zeros_like(emu_spec[:, 0, :])

        # If both tracers are galaxies, Pk^{tr1,tr2} = f_i^bin1 * f_j^bin2 * Pk_ij
        if 'gc' in clm['bin_1'] and 'gc' in clm['bin_2']:
            bias_eft1 = b_trs[clm['bin_1']]
            bias_eft2 = b_trs[clm['bin_2']]

            for key1 in bias_eft1.keys():
                bias1 = bias_eft1[key1]

                for key2 in bias_eft2.keys():
                    bias2 = bias_eft2[key2]

                    if key1+'_'+key2 in self.emu_dict.keys():
                        comb = self.emu_dict[key1+'_'+key2]
                    else:
                        comb = self.emu_dict[key2+'_'+key1]
                    Pk_a += bias1*bias2*emu_spec[:, comb, :]

        # If first tracer is galaxies, Pk^{tr1,tr2} = f_i^bin1 * 1. * Pk_0i
        elif 'gc' in clm['bin_1'] and 'wl' in clm['bin_2']:
            bias_eft1 = b_trs[clm['bin_1']]

            for key1 in bias_eft1.keys():
                bias1 = bias_eft1[key1]
                comb = self.emu_dict['b0'+'_'+key1]
                Pk_a += bias1*emu_spec[:, comb, :]

        # If second tracer is galaxies, Pk^{tr1,tr2} = f_j^bin2 * 1. * Pk_0j
        elif 'wl' in clm['bin_1'] and 'gc' in clm['bin_2']:
            bias_eft2 = b_trs[clm['bin_2']]

            for key2 in bias_eft2.keys():
                bias2 = bias_eft2[key2]
                comb = self.emu_dict['b0'+'_'+key2]
                Pk_a += bias2*emu_spec[:, comb, :]

        # Convert ks from [Mpc/h]^-1 to [Mpc]^-1
        lk_arr = np.log(self.k*cosmo['H0']/100.)
                          
        # Same for the power spectrum: convert to Mpc^3
        Pk_a /= (cosmo['H0']/100.)**3.
        
        # Compute the 2D power spectrum
        p_of_k_a = ccl.Pk2D(a_arr=self.a, lk_arr=lk_arr, pk_arr=Pk_a, is_logp=False)
        
        return p_of_k_a

    def _get_b_heft(self, tn, **pars):
        # Read in the 4 HEFT parameters (b1, b2, bs, bn) and enforce b0 = 1.
        b_heft = {}
        for b in self.b_emu:
            b_heft[b] = pars.get(self.input_params_prefix + '_' + tn + '_' + b, 0.)
        # B.H. cheating (move to params)
        #b_heft['b0'] = 1. # TESTING
        #assert b_heft['b0'] == 1., "If using the HEFT model, b0 needs to be set to 1"
        return b_heft
        
    def _get_cl_wl(self, cosmo, pk, **pars):

        # Compute all C_ells without multiplicative bias
        trs = {}
        b_trs = {} # only used for HEFT
        for tn in self.used_tracers:
            trs[tn] = self._get_tracer(cosmo, tn, **pars)
            if self.bz_model == 'HEFT' and 'wl' not in tn:
                b_trs[tn] = self._get_b_heft(tn, **pars)

        # Compute the power spectrum for the HEFT model
        if self.bz_model == 'HEFT':
            emu_spec = self._compute_emu_spec(cosmo)
                
        cls = []
        for clm in self.cl_meta:
            # Compute the angular power spectrum with more care if using HEFT model
            if self.bz_model == 'HEFT':
                if 'wl' in clm['bin_1'] and 'wl' in clm['bin_2']:
                    cl = ccl.angular_cl(cosmo, trs[clm['bin_1']], trs[clm['bin_2']],
                                        self.l_sample, p_of_k_a=pk)
                else:
                    # Obtain power spectrum as a function of k and a
                    p_of_k_a = self._get_p_of_k_a(emu_spec, cosmo, b_trs, clm)
                    cl = ccl.angular_cl(cosmo, trs[clm['bin_1']], trs[clm['bin_2']],
                                        self.l_sample, p_of_k_a=p_of_k_a)
            else:
                cl = ccl.angular_cl(cosmo, trs[clm['bin_1']], trs[clm['bin_2']],
                                    self.l_sample, p_of_k_a=pk)
            clb = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])
            cls.append(clb)
        return cls

    def _get_theory(self, **pars):
        # Compute theory vector
        res = self.provider.get_CCL()
        cosmo = res['cosmo']
        pk = res['pk']
        cls = self._get_cl_wl(cosmo, pk, **pars)
        cl_out = np.zeros(self.ndata)
        for clm, cl in zip(self.cl_meta, cls):
            if 'wl' in clm['bin_1']:
                m1 = pars[self.input_params_prefix + '_' + clm['bin_1'] + '_m']
            else:
                m1 = 0.
            if 'wl' in clm['bin_2']:
                m2 = pars[self.input_params_prefix + '_' + clm['bin_2'] + '_m']
            else:
                m2 = 0.
            prefac = (1+m1) * (1+m2)
            cl_out[clm['inds']] = cl * prefac
        return cl_out

    def get_requirements(self):
        # By selecting `self._get_pk` as a `method` of CCL here,
        # we make sure that this function is only run when the
        # cosmological parameters vary.
        return {'CCL': {'methods': {'pk': self._get_pk}}}

    def logp(self, **pars):
        """
        Simple Gaussian likelihood.
        """
        t = self._get_theory(**pars)
        r = t - self.data_vec
        chi2 = np.dot(r, self.inv_cov.dot(r))
        return -0.5*chi2
