# Step 2: Inject sets of transits into the original lightcurves

import numpy as np
import os

import juliet
import batman
import exoctk
import glob

import ray

ray.shutdown()
ray.init()

# Read in the list of planets
planets, tic_ids = np.genfromtxt('Final_List.dat', dtype = (str,str), delimiter = '\t', unpack = True)
print("Imported list of " + str(len(planets)) + " targets")

repeats = 100 # injecting 100 sets
GPmodel = 'QP'

def get_transit_model(n, sector, planet, GPmodel): 
    paramsfile = glob.glob(planet + '/' + sector + '_' + GPmodel + '_in_transit*' + '/posteriors.dat')
    paramsfile_priors = glob.glob(planet + '/' + sector + '_' + GPmodel + '_in_transit*' + '/priors.dat')
    # The file path is paramsfile[0]
    
    # Read in the data: 
    median, upper68, lower68 = np.genfromtxt(paramsfile[0], usecols = (1, 2, 3), unpack = True)
    dist, value = np.genfromtxt(paramsfile_priors[0], usecols = (1, 2), dtype = str, unpack = True)
    
    params = batman.TransitParams()
    
    # Extract these from the previous sector-wide fits
    params.t0 = 0   # time of inferior conjunction OR  2459782.0096445102
    params.per = median[6]  # orbital period (days)
    params.a = median[12]    # semi-major axis (in units of stellar radii)
    params.rp = median[8]   # rp/rs
    params.ecc = float(value[13])   # eccentricity
    params.w = float(value[14])    # longitude of periastron (in degrees) p
    params.inc = np.degrees(np.arccos(median[9] / median[12]))  # orbital inclination (in degrees)
    params.limb_dark = 'quadratic' # limb darkening profile to use
    params.u = [median[10], median[11]] # limb darkening coefficients
    t = np.linspace(- 1 * n / 1440., 1 * n / 1440., n)
    
    tmodel = batman.TransitModel(params, t.astype('float64'))
    return t, tmodel.light_curve(params)

@ray.remote
def inject(tdur, repeats, planet, sector, repeat, expected, t0, t_TESS, f_TESS, ferr_TESS, GPmodel, sector_keys): 
    number = 0
    
    file = glob.glob(planet + '/' + sector + '_' + GPmodel + '_in_transit*' + '/posteriors.dat')
    med, u68, l68 = np.genfromtxt(file[0], usecols = (1, 2, 3), unpack = True)
    per = med[6]
    
    # Use these for calculating the offset
    offset = 2. * tdur + repeat * (per - 4. * tdur) / (repeats - 1.0) # Different offset for each iteration
    n = round(tdur * 1440.)
    
    repeat_folder = planet + "/" + str(sector) + "/run_" + str(repeat)
    if not os.path.exists(repeat_folder): 
        return # No need to run, has already been removed
    
    tt_all = []
    injected_all = []
    fferr_all = []
    t0_injected = []

    for transit in range(0, int(27.4 / per) + 2): # Transits per sector
        if number == expected: # There are now enough transits; no need to inject more
            break
        
        # Create dataset: 
        tt = []
        injected = []
        fferr = []

        # Set number of datapoints the lightcurve will have; define dictionaries that will host the data:
        times, fluxes = {}, {}

        # Get data and put it in the dictionaries
        times['TESS'], fluxes['TESS'] = get_transit_model(n, sector, planet, GPmodel)

        # Now offset the data: 
        t0_current = t0 + offset + transit * per
        times['TESS'] = times['TESS'] + t0_current

        firsttime = True
        for time in range(len(t_TESS)): 
            counter = False
            for sim_time in range(len(times['TESS'])): 
                if abs(times['TESS'][sim_time] - t_TESS[time]) <= 1.0/1440: 
                    counter = True # There is a data point that is within 1 minute of the TESS data point
                    break
            if counter == True: 
                injected.append(f_TESS[time] * fluxes['TESS'][sim_time])
                tt.append(t_TESS[time])
                fferr.append(ferr_TESS[time])

                if firsttime == True: 
                    t0_injected.append(t0_current)
                    firsttime = False
        
        if (len(injected) / n <= 1.05) and (len(injected) / n >= 0.95): # 5% error allowed on number of data points
            number += 1

            outfile = repeat_folder + "/transit_" + str(transit) + ".dat"

            file = open(outfile, 'w')
            for i in range(0, len(injected)): 
                file.write(format(tt[i], '.10f') + " " + format(injected[i], '.10f') + " " + format(fferr[i], '.10f') + " TESS")
                file.write("\n")
            file.close()
        else: # This means that the transit that we just went through is invalid
            t0_injected = t0_injected[:-1]

        # Write out the combined lightcurves
        tt_all += tt
        injected_all += injected
        fferr_all += fferr
        #print('Run: ' + str(repeat) + ' | Injected', number, 'transits')
    
    # If the number of transits isn't as expected, "comment out" the folder
    if (number / expected > 1.2) or (number / expected < 0.8): # Remove if difference > 20%
        for key in sector_keys: 
            os.rename(planet + "/" + str(key) + "/run_" + str(repeat), planet + "/" + str(key) + "/#run_" + str(repeat))
    
    if repeat == repeats - 1: # Final set
        print("Finished " + sector + " for " + planet)

# Main code starts here
for planet in planets: 
    print("Getting data for: " + str(planet))
    t, f, ferr = juliet.get_all_TESS_data(planet)
    
    bad_sectors = [] # Removing bad sectors consistently
    
    for sector in t.keys(): 
        if len(glob.glob(planet + '/' + sector + '_' + GPmodel + '_in_transit*')) == 0: 
            bad_sectors.append(sector)
    
    if len(bad_sectors) > 0: # Remove bad sectors
        for bad_sector in bad_sectors: 
            del t[bad_sector]
            del f[bad_sector]
            del ferr[bad_sector]
    
    nsectors = len(list(t.keys()))
    
    planet_data, url = exoctk.utils.get_target_data(planet)
    tdur = planet_data['transit_duration']
    
    for sector in t.keys(): 
        t_TESS, f_TESS, ferr_TESS = t[sector], f[sector], ferr[sector]
        
        if not os.path.exists(planet + "/" + str(sector)): 
            os.mkdir(planet + "/" + str(sector))
        
        injected = []
        
        # Count the number of good transits
        transit_folders = glob.glob(planet + '/transit_*')
        
        expected = 0
        for transit_folder in transit_folders: 
            if transit_folder.split('_')[-1] == sector: 
                param1, upp1, low1 = np.genfromtxt(transit_folder + '/posteriors.dat', usecols = (1, 2, 3), unpack = True)
                err1 = (upp1[8] + low1[8]) / 2.0
                if (param1[8] / err1) < 5.0: # "Comment out" transits with SNR < 5
                    os.rename(transit_folder, planet + '/#transit' + transit_folder.split('/transit')[-1]) # Transit has low SNR and will not be counted
                else: 
                    expected += 1
        print("Expected number of transits: " + str(expected))
        
        t0_file = glob.glob(planet + '/' + sector + '_' + GPmodel + '_in_transit*' + '/posteriors.dat')
        
        param, upp, low = np.genfromtxt(t0_file[0], usecols = (1, 2, 3), unpack = True)
        t0 = param[7] - param[6] # Subtract one period from t0
        
        # Make the folders first
        for repeat in range(0, repeats): 
            run_folder = planet + "/" + str(sector) + "/run_" + str(repeat)
            os.mkdir(run_folder)
        
        # Then inject the transits
        for repeat in range(0, repeats): 
            inject.remote(tdur, repeats, planet, sector, repeat, expected, t0, t_TESS, f_TESS, ferr_TESS, GPmodel, t.keys())
