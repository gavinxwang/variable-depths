# Step 3: Fit all sets of simulated transits

# Import python libs:
import numpy as np
import os

# Import auxiliary libraries:
import juliet
import exoctk
import matplotlib.pyplot as plt

# Import libraries for this script:
import utils # Modified utils package for fitting 

folder = '' # Parent directory of folder with planet name
dataname = 'Final_List.dat'
datafile = folder + dataname

# Exposure time of 2-min cadence data, in days:
exp_time = (2./60.)/24.

# Load target list. This will only load the planet name and TIC IDs
target_list = utils.read_data(datafile)

# The factor variable defines what is "in-transit" in units of the transit duration:
factor = 2.0

# Choose GP Model: ExpMatern, Matern, QP
GPmodel = 'QP'

# Iterate through all planets:
for planet in target_list.keys():
    print('Working on '+ planet)
    
    # Load all data for this particular planet; if data exists, go forward. If not, skip
    try:
        planet_data, url = exoctk.utils.get_target_data(planet)
        
        # Extract useful data:
        tdur = planet_data['transit_duration']
        tdepth = planet_data['transit_depth']
        period = planet_data['orbital_period']
        period_err = (planet_data['orbital_period_upper'] + planet_data['orbital_period_lower'])*0.5
        t0 = planet_data['transit_time'] + 2400000.5
        t0_err = (planet_data['transit_time_upper'] + planet_data['transit_time_lower'])*0.5
        
        # If data is not float (e.g., empty values), reject system:
        if (type(tdur) is float) and (type(tdepth) is float) and (type(period) is float) and (type(period_err) is float) and \
           (type(t0) is float) and (type(t0_err) is float):
            has_data = True
        else:
            print('Something is wrong with ' + planet +' data. Skipping.')
            has_data = False

        # Now check eccentricity and omega. If no data, set to 0 and 90:
        ecc, omega = planet_data['eccentricity'], planet_data['omega']
        if (type(ecc) is not float or type(omega) is not float):
            ecc = 0.
            omega = 90.
    except: 
        print('No planetary data for ' + planet)
        has_data = False

    # If it has data, we move ahead with the analysis:
    if has_data: 
        # Load data for each sector:
        t, f, ferr = juliet.utils.get_all_TESS_data('TIC ' + target_list[planet]['ticid'])
        
        utils.fit_transit_by_transit(period, period_err, t0, t0_err, ecc, omega, GPmodel = GPmodel, outpath = planet, in_transit_length = factor*tdur)
