import numpy as np
import glob
import pickle
import juliet
import os

def fit(t, f, ferr, sector, P, P_err, t0, t0_err, ecc, omega, GPmodel = 'QP', outpath = 'planetfit', method = '', in_transit_length = 0., fit_catwoman = False):

    # Scale t0 to the transit closest to the center of the TESS observations:
    # n = int((np.mean(t) - t0)/P)
    
    t0 -= 1000. * P # First subtract from t0 so as to move the "cursor" to before the first transit
    n = int((np.min(t) - t0) / P) + 1 # New t0 that corresponds to the first transit of the sector
    t0 += n*P
    
    t0_err = np.sqrt(t0_err**2 + (n * P_err)**2)
    
    # Define priors:
    priors = {}

    # First define parameter names, distributions and hyperparameters for GP-independant parameters:
    if not fit_catwoman:

        params1 = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_TESS', 'q2_TESS', \
                   'ecc_p1', 'omega_p1', 'a_p1']
        
        params1_instrument = ['mdilution_TESS', 'mflux_TESS', 'sigma_w_TESS']

        dists1 = ['normal', 'normal', 'uniform', 'uniform', 'uniform', 'uniform', \
                   'fixed','fixed','loguniform']
        
        dists1_instrument = ['fixed','normal','loguniform']

        hyperps1 = [[P,P_err], [t0, 0.1], [0., 1.], [0., 2.], [0., 1.], [0., 1.], \
                   ecc, omega, [1., 100.]]

    else:

        params1 = ['P_p1', 't0_p1', 'p1_p1', 'p2_p1', 'phi_p1', 'b_p1', 'q1_TESS', 'q2_TESS', \
                   'ecc_p1', 'omega_p1', 'a_p1']
        
        params1_instrument = ['mdilution_TESS', 'mflux_TESS', 'sigma_w_TESS']

        dists1 = ['normal', 'normal', 'uniform', 'uniform', 'fixed', 'uniform', 'uniform', 'uniform', \
                   'fixed','fixed','loguniform']
        
        dists1_instrument = ['fixed','normal','loguniform']

        hyperps1 = [[P,P_err], [t0, 0.1], [0., 1.], [0., 1.], 90., [0., 2.], [0., 1.], [0., 1.], \
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

    # If method is blank, fit both simultaneously. If set to "fit_out", fit out-of-transit lightcurve first, use posteriors of that 
    # fit as priors to an in-transit fit. The in_transit_length measures in days what is "in-transit", centered around t0:
    if method == '':
        params = params1 + params1_instrument + params2
        dists = dists1 + dists1_instrument + dists2
        hyperps = hyperps1 + hyperps1_instrument + hyperps2

        # Populate the priors dictionary:
        for param, dist, hyperp in zip(params, dists, hyperps):
            priors[param] = {}
            priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

        # Port data in the juliet format:
        tt, ff, fferr = {}, {}, {}
        tt['TESS'], ff['TESS'], fferr['TESS'] = t, f, ferr

        # Run fit:
        dataset = juliet.load(priors=priors, t_lc = tt, y_lc = ff, \
                              yerr_lc = fferr, GP_regressors_lc = tt, out_folder = outpath+'_'+GPmodel)
        results = dataset.fit(sampler = 'dynesty', nthreads = 16)
    else:
        # Peform GP fit first:
        params = params1_instrument + params2
        dists = dists1_instrument + dists2
        hyperps = hyperps1_instrument + hyperps2

        # Populate priors dict:
        for param, dist, hyperp in zip(params, dists, hyperps):
            priors[param] = {}
            priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

        # Select only out-of-transit data. For this, work on phase-space:
        phases = juliet.utils.get_phases(t, P, t0)
        idx_oot = np.where(np.abs(phases*P) >= in_transit_length*0.5)[0]

        # Save data dict:
        tt, ff, fferr = {}, {}, {}
        tt['TESS'], ff['TESS'], fferr['TESS'] = t[idx_oot], f[idx_oot], ferr[idx_oot]

        # Run GP-only fit:
        dataset = juliet.load(priors=priors, t_lc = tt, y_lc = ff, \
                              yerr_lc = fferr, GP_regressors_lc = tt, out_folder = outpath+'_'+GPmodel+'_out_of_transit')
        results = dataset.fit(sampler = 'dynesty', nthreads = 16)

        # Now use posteriors of that fit to fit the in-transit data. Assume truncated normals for the GP hyperparameters:
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

        # And with those changes, fit the in-transit data:
        idx_in = np.where(np.abs(phases*P) < in_transit_length*0.5)[0]

        # Save data dict:
        tt['TESS'], ff['TESS'], fferr['TESS'] = t[idx_in], f[idx_in], ferr[idx_in]

        # Run fit:
        if not fit_catwoman:

            dataset = juliet.load(priors=priors, t_lc = tt, y_lc = ff, \
                    yerr_lc = fferr, GP_regressors_lc = tt, out_folder = outpath+'_'+GPmodel+'_in_transit_batman')

        else:

            dataset = juliet.load(priors=priors, t_lc = tt, y_lc = ff, \
                    yerr_lc = fferr, GP_regressors_lc = tt, out_folder = outpath+'_'+GPmodel+'_in_transit_catwoman')

        results = dataset.fit(sampler = 'dynesty', nthreads = 16)

def fit_transit_by_transit(P, P_err, t0, t0_err, ecc, omega, GPmodel = 'QP', outpath = 'planetfit', in_transit_length = 0.):

    # First, extract both sectors and folders of those sectors which have out-of-transit fits already done:
    oot_folders = glob.glob(outpath+'/TESS*_'+GPmodel+'_out_of_transit')

    for oot_folder in oot_folders:
        print('Working on',oot_folder)
        it_folder = oot_folder.split('out_of_transit')[0]+'in_transit_batman' 
        
        sector = oot_folder.replace(outpath + '/', '')
        sector = sector.replace('_'+GPmodel+'_out_of_transit', '')
        print("Sector: " + sector)
        
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
        results = dataset.fit(sampler = 'dynesty', nthreads = 16)

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

        # Now extract in-transit data from in-transit fit to sector:
        dataset = juliet.load(input_folder = it_folder)

        # Iterate through each of the transits in the sector:
        idx = np.where(np.abs(np.diff(dataset.t_lc))>0.5)[0]
        idx = np.append(idx, len(dataset.t_lc)) # Add to fit the last transit; otherwise, the last transit is always missing
        # print(idx)
        start_idx = -1

        print('Detected',len(idx),'transits')
        
        # os.rename(it_folder, it_folder + '_' + str(len(idx))) # Append estimate of total number of transits to folders
        
        for i in idx:
            tt, ff, fferr = {}, {}, {}
            tt['TESS'], ff['TESS'], fferr['TESS'] = dataset.t_lc[start_idx+1:i], dataset.y_lc[start_idx+1:i], dataset.yerr_lc[start_idx+1:i]

            # Guess which t0 this dataset corresponds to:
            mid_idx = int(len(tt['TESS'])*0.5)
            tmid = tt['TESS'][mid_idx]
            n = (tmid-t0)/P
            tc = t0 + n*P
            
            # Check if there is any time-datapoint that covers, at least, an hour around mid-transit:
            n_onehour = len(np.where(np.abs(tt['TESS']-tc)<1./24.)[0])

            # If there are datapoints, fit the dataset. Use that central time as the t0 mean on the prior:
            if n_onehour > 0:
                priors['t0_p1']['hyperparameters'][0] = tc # tc is the value of the transit that I want

                # Run fit:
                transit_dataset = juliet.load(priors=priors, t_lc = tt, y_lc = ff, \
                          yerr_lc = fferr, GP_regressors_lc = tt, out_folder = outpath+'/transit_'+str(n)+'_'+GPmodel+'_in_transit_batman_'+sector)
                results = transit_dataset.fit(sampler = 'dynesty', nthreads = 16)
            else:
                print('Transit at',tc,' doesnt have n_onehour apparently:',np.abs(tt['TESS']-tc))
            start_idx = i

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
