import numpy as np
import glob
import pickle
import juliet
import matplotlib.pyplot as plt

import ray

ray.shutdown()
ray.init()

def fit_transit_by_transit(P, P_err, t0, t0_err, ecc, omega, GPmodel = 'QP', outpath = 'planetfit', in_transit_length = 0.):

    # First, extract both sectors and folders of those sectors which have out-of-transit fits already done:
    oot_folders = glob.glob(outpath+'/TESS*_'+GPmodel+'_out_of_transit')

    for oot_folder in oot_folders:
        print('Working on',oot_folder)

        # Define priors:
        priors = {}

        # First define parameter names, distributions and hyperparameters for GP-independant parameters:
        params1 = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_TESS', 'q2_TESS', \
                   'ecc_p1', 'omega_p1', 'a_p1']

        params1_instrument = ['mdilution_TESS', 'mflux_TESS', 'sigma_w_TESS']

        dists1 = ['normal', 'normal', 'uniform', 'uniform', 'uniform', 'uniform', \
                   'fixed','fixed','loguniform']

        dists1_instrument = ['fixed','normal','loguniform']

        hyperps1 = [[P,P_err], [t0, 0.1], [0., 1.], [0., 1.], [0., 1.], [0., 1.], \
                   ecc, omega, [1., 100.]]

        hyperps1_instrument = [1., [0., 0.1], [0.1, 10000.]]

        # Now define hyperparameters of the GP depending on the chosen kernel:
        if GPmodel == 'ExpMatern':
            params2 = ['GP_sigma_TESS', 'GP_timescale_TESS', 'GP_rho_TESS']
            dists2 = ['loguniform', 'loguniform', 'loguniform']
            hyperps2 = [[1e-5, 10000.], [1e-3,1e2], [1e-3,1e2]]
        elif GPmodel == 'Matern':
            params2 = ['GP_sigma_TESS', 'GP_rho_TESS']
            dists2 = ['loguniform', 'loguniform']
            hyperps2 = [[1e-5, 10000.], [1e-3,1e2]]
        elif GPmodel == 'QP':
            params2 = ['GP_B_TESS', 'GP_C_TESS', 'GP_L_TESS', 'GP_Prot_TESS']
            dists2 = ['loguniform', 'loguniform', 'loguniform','loguniform']
            hyperps2 = [[1e-5,1e3], [1e-5,1e4], [1e-3, 1e3], [1.,1e2]]

        # Extract posteriors from out-of-transit GP fit first:
        params = params1_instrument + params2
        dists = dists1_instrument + dists2
        hyperps = hyperps1_instrument + hyperps2

        # Populate priors dict:
        for param, dist, hyperp in zip(params, dists, hyperps):
            priors[param] = {}
            priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

        dataset = juliet.load(input_folder = oot_folder)
        results = dataset.fit(sampler = 'dynesty', nthreads = 12)

        for i in range(len(params2)):
            posterior = results.posteriors['posterior_samples'][params2[i]]
            mu, sigma = np.median(posterior), np.sqrt(np.var(posterior))
            dists2[i] = 'truncatednormal'
            hyperps2[i] = [mu, sigma, hyperps2[i][0], hyperps2[i][1]]

        # Same for sigma_w and mflux:
        dists1_instrument[2] = 'truncatednormal'
        posterior = results.posteriors['posterior_samples']['sigma_w_TESS']
        mu, sigma = np.median(posterior), np.sqrt(np.var(posterior))
        hyperps1_instrument[2] = [mu, sigma, hyperps1_instrument[2][0], hyperps1_instrument[2][1]]

        # Normal for mflux:
        dists1_instrument[1] = 'normal'
        posterior = results.posteriors['posterior_samples']['mflux_TESS']
        mu, sigma = np.median(posterior), np.sqrt(np.var(posterior))
        hyperps1_instrument[1] = [mu, sigma]

        # Populate prior dict:
        params = params1 + params1_instrument + params2
        dists = dists1 + dists1_instrument + dists2
        hyperps = hyperps1 + hyperps1_instrument + hyperps2

        # Populate the priors dictionary:
        for param, dist, hyperp in zip(params, dists, hyperps):
            priors[param] = {}
            priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

        it_folders = glob.glob(outpath+'/TESS*'+'/run_*/')
        
        #if len(it_folders) / len(oot_folders) < 10.: # Not enough successful injections -- don't fit
        #    continue
        
        for it_folder in it_folders: 
            run_id = 0
            it_files = glob.glob(it_folder+'/transit_*'+'.dat')

            fit_injected_transit.remote(P, t0, priors, it_files, run_id) # Call the remote function
            run_id += 1

@ray.remote
def fit_injected_transit(P, t0, priors, it_files, run_id): 
    for it_file in it_files: 
        tt, ff, fferr = {}, {}, {}
        tt['TESS'], ff['TESS'], fferr['TESS'] = np.genfromtxt(it_file, usecols = (0, 1, 2), unpack = True)

        mid_idx = int(len(tt['TESS'])*0.5)
        tmid = tt['TESS'][mid_idx]

        # Check if there is any time-datapoint that covers, at least, an hour around mid-transit:
        n_onehour = len(np.where(np.abs(tt['TESS']-tmid)<1./24.)[0])
        
        # If there are datapoints, fit the dataset. Use that central time as the t0 mean on the prior:
        if n_onehour > 0:
            priors['t0_p1']['hyperparameters'][0] = tmid
            print(it_file[:-4])

            # Run fit:
            transit_dataset = juliet.load(priors=priors, t_lc = tt, y_lc = ff, yerr_lc = fferr, GP_regressors_lc = tt, \
                                          out_folder = it_file[:-4])
            
            results = transit_dataset.fit(sampler = 'dynesty')
        else:
            print('Transit at',tc,' doesnt have n_onehour apparently:',np.abs(tt['TESS']-tc))

def read_data(fname):
    fin = open(fname, 'r')
    data = {}
    while True:
        line = fin.readline()
        line = line[:-1] # Remove the trailing "\n"
        if line != '':
            if line[0] != '#':
                lv = line.split("\t")
                name, ticid = lv[0], lv[1]
                data[name] = {}
                data[name]['ticid'] = ticid
        else:
            break
    return data
