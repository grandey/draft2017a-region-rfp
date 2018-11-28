"""
Functions for draft2017a-region-rfp.

Warning:
    The complete functionality of these functions has not been extensively tested.
    They have only been tested for the specific purposes used in the draft2017a analysis.

Dependencies:
    - climapy (https://github.com/grandey/climapy, https://doi.org/10.5281/zenodo.1053020)
    - pandas
    - xarray

Data requirements:
    - Default MAM3 DMS emissions data (aerocom_mam3_dms_surf_2000_c090129.nc)
    - Modified MAM3 emissions data, archived at https://doi.org/10.6084/m9.figshare.6972827
    - CESM output data in timeseries format, archived at
      https://doi.org/10.6084/m9.figshare.6972827


Author:
    Benjamin S. Grandey, 2017-2018
"""

import climapy  # https://doi.org/10.5281/zenodo.1053020
import numpy as np
import os
import pandas as pd
from scipy.stats import ttest_ind
import xarray as xr


# Directories holding data
dms_filename = os.path.expandvars('$HOME/data/inputdataCESM/trop_mozart_aero/emis/'
                                  'aerocom_mam3_dms_surf_2000_c090129.nc')  # default DMS file
emissions_dir = os.path.expandvars('$HOME/data/figshare/figshare6972827/'
                                   'p16a_F/')  # https://doi.org/10.6084/m9.figshare.6972827
output_dir = os.path.expandvars('$HOME/data/figshare/figshare6972827/')
#emissions_dir = os.path.expandvars('$HOME/data/projects/p2016a_hist_reg/input/'
#                                   'p16a_F/')  # excluding DMS
#output_dir = os.path.expandvars('$HOME/data/drafts/draft2017a_region_rfp_data/'
#                                'output_timeseries/')


def dependency_versions():
    """
    Get versions of dependencies.
    """
    version_dict = {}
    for package in [climapy, np, os, pd, xr]:
        try:
            version_dict[package.__name__] = package.__version__
        except AttributeError:
            pass
    return version_dict


def load_region_bounds_dict():
    """
    Load dictionary containing region short names (keys) and bounds (values).
    """
    region_bounds_dict = {'EAs': [(94, 156), (20, 65)],  # longitude tuple, latitude tuple
                          'SEAs': [(94, 161), (-10, 20)],
                          'ANZ': [(109, 179), (-50, -10)],
                          'SAs': [(61, 94), (0, 35)],
                          'AfME': [(-21, 61), (-40, 35)],
                          'Eur': [(-26, 31), (35, 75)],
                          'CAs': [(31, 94), (35, 75)],
                          'NAm': [(-169, -51), (15, 75)],
                          'SAm': [(-94, -31), (-60, 15)],
                          'Globe': [None, None]}
    return region_bounds_dict


_region_bounds_dict = load_region_bounds_dict()  # for internal use


def load_region_long_dict():
    """
    Load dictionary containing region short names (keys) and long names (values).
    """
    region_long_dict = {'EAs': 'East Asia',
                        'SEAs': 'Southeast Asia',
                        'ANZ': 'Australia and New Zealand',
                        'SAs': 'South Asia',
                        'AfME': 'Africa and the Middle East',
                        'Eur': 'Europe',
                        'CAs': 'Central Asia',
                        'NAm': 'North America',
                        'SAm': 'South America',
                        'Globe': 'globe'}
    return region_long_dict


_region_long_dict = load_region_long_dict()


def load_scenario_name_dict():
    """
    Load dictionary containing scenarios and the names used to refer to the scenarios.
    """
    scenario_name_dict = {'p16a_F_Hist_2000': 'All1',
                          'p16a_F_Zero_2000': 'All0',
                          'p17b_F_Hist_2000': 'Correct1'}
    for region in list(load_region_bounds_dict().keys()) + ['Ship', ]:
        if region != 'Globe':
            scenario_name_dict['p16a_F_No{}_2000'.format(region)] = '{}0'.format(region)
            scenario_name_dict['p16a_F_Only{}_2000'.format(region)] = '{}1'.format(region)
    return scenario_name_dict


_scenario_name_dict = load_scenario_name_dict()
_inverted_scenario_name_dict = {v: k for k, v in _scenario_name_dict.items()}


def load_variable_long_dict():
    """
    Load dictionary containing variable long names of CESM output variables of potential interest,
    assuming one is looking at differences between scenarios.
    """
    variable_long_dict = {'FSNTOA+LWCF': 'Net Effective Radiative Forcing',
                          'SWCF_d1': r'$\Delta$ Clean-sky Shortwave Cloud Radiative Effect',
                          'LWCF': r'$\Delta$ Longwave Cloud Radiative Effect',
                          'FSNTOA-FSNTOA_d1': r'$\Delta$ Direct Radiative Effect',
                          'FSNTOAC_d1': r'$\Delta$ Surface Albedo Radiative Effect',
                          'BURDENSO4': r'$\Delta$ Sulfate Aerosol Burden',
                          'BURDENPOM': r'$\Delta$ Primary Organic Matter Aerosol Burden',
                          'BURDENBC': r'$\Delta$ Black Carbon Aerosol Burden',
                          'TGCLDIWP': r'$\Delta$ Grid-box Ice Water Path',
                          'TGCLDLWP': r'$\Delta$ Grid-box Liquid Water Path',
                          'CCN3_ml24': r'$\Delta$ CCN Concentration at 0.1% Supersaturation ' +
                                       'in Model Level 24',
                          'CDNUMC': r'$\Delta$ Column-integrated Cloud Droplet Number ' +
                                    'Concentration',
                          'AEROD_v': r'$\Delta$ Aerosol Optical Depth'}
    return variable_long_dict


_variable_long_dict = load_variable_long_dict()


def load_variable_symbol_dict():
    """
    Load dictionary containing variable symbols, for differences between scenarios.
    """
    variable_symbol_dict = {'FSNTOA+LWCF': r'$ERF_\mathrm{SW+LW}$',
                            'SWCF_d1': r'$\Delta CRE_\mathrm{SW}$',
                            'LWCF': r'$\Delta CRE_\mathrm{LW}$',
                            'FSNTOA-FSNTOA_d1': r'$\Delta DRE_\mathrm{SW}$',
                            'FSNTOAC_d1': r'$\Delta SRE_\mathrm{SW}$',
                            'BURDENSO4': r'$\Delta Burden_\mathrm{SO4}$',
                            'BURDENPOM': r'$\Delta Burden_\mathrm{POM}$',
                            'BURDENBC': r'$\Delta Burden_\mathrm{BC}$',
                            'TGCLDIWP': r'$\Delta WP_\mathrm{ice}$',
                            'TGCLDLWP': r'$\Delta WP_\mathrm{liquid}$',
                            'CCN3_ml24': r'$\Delta CCN_\mathrm{conc,lev24}$',
                            'CDNUMC': r'$\Delta CDNC_\mathrm{column}$',
                            'AEROD_v': r'$\Delta AOD$'}
    return variable_symbol_dict


_variable_symbol_dict = load_variable_symbol_dict()


def load_variable_units_dict():
    """
    Load dictionary containing variable units - after scale factors have been applied.
    """
    variable_units_dict = {'FSNTOA+LWCF': r'W m$^{-2}$',
                           'SWCF_d1': r'W m$^{-2}$',
                           'LWCF': r'W m$^{-2}$',
                           'FSNTOA-FSNTOA_d1': r'W m$^{-2}$',
                           'FSNTOAC_d1': r'W m$^{-2}$',
                           'BURDENSO4': r'mg(SO$_4$) m$^{-2}$',
                           'BURDENPOM': r'mg m$^{-2}$',
                           'BURDENBC': r'mg m$^{-2}$',
                           'TGCLDIWP': r'g m$^{-2}$',
                           'TGCLDLWP': r'g m$^{-2}$',
                           'CCN3_ml24': r'cm$^{-3}$',
                           'CDNUMC': r'×$10^6$ cm$^{-2}$',
                           'AEROD_v': None}
    return variable_units_dict


_variable_units_dict = load_variable_units_dict()


def load_variable_sf_dict():
    """
    Load dictionary containing scale-factors to apply to variables.
    """
    variable_sf_dict = {'BURDENSO4': 1e6,  # kg/m2 to mg/m2
                        'BURDENPOM': 1e6,  # kg/m2 to mg/m2
                        'BURDENBC': 1e6,  # kg/m2 to mg/m2
                        'TGCLDIWP': 1e3,  # kg/m2 to g/m2
                        'TGCLDLWP': 1e3,  # kg/m2 to g/m2
                        'CDNUMC': 1e-10,  # /m2 to ×10^6/cm2
                        }
    return variable_sf_dict


_variable_sf_dict = load_variable_sf_dict()


def load_species_sf_dict():
    """
    Load dictionary of scale factors for molecules/cm2/s -> g/m2/yr and
    particles/cm2/s*6.022e26 -> particles/m2/yr for different emitted species
    """
    species_sf_dict = {'so2': (365*24*60*60)*(100*100)*(32/6.02214e23),  # g(S)/m2/yr
                       'oc': (365*24*60*60)*(100*100)*(12/6.02214e23),
                       'bc': (365*24*60*60)*(100*100)*(12/6.02214e23),
                       'so4_a1': (365*24*60*60)*(100*100)*(32/6.02214e23),  # g(S)/m2/yr
                       'num_a1': (365*24*60*60)*(100*100)*(1/6.022e26),  # particles/m2/yr
                       'so4_a2': (365*24*60*60)*(100*100)*(32/6.02214e23),
                       'num_a2': (365*24*60*60)*(100*100)*(1/6.022e26),  # particles/m2/yr
                       'dms': (365*24*60*60)*(100*100)*(32/6.02214e23)}  # g(S)/m2/yr
    return species_sf_dict


_species_sf_dict = load_species_sf_dict()


def load_emissions(species='so2', surf_or_elev='both', scenario='Hist_2000', season='annual'):
    """
    Load annual/seasonal emissions for a specific species and scenario.

    Args:
        species: species name (default 'so2')
        surf_or_elev: 'surf' (surface), 'elev' (elevated), or 'both' (sum; default)
        scenario: scenario name (default 'Hist_2000')
        season: 'annual' (default) or name of season (e.g 'DJF')

    Returns:
        xarray DataArray
    """
    # If DMS, emissions are all at the surface
    if species == 'dms' and surf_or_elev == 'both':
        data = load_emissions(species='dms', surf_or_elev='surf', scenario=scenario, season=season)
    # If 'both', call function recursively
    elif surf_or_elev == 'both':
        surf_data = load_emissions(species=species, surf_or_elev='surf', scenario=scenario,
                                   season=season)
        elev_data = load_emissions(species=species, surf_or_elev='elev', scenario=scenario,
                                   season=season)
        data = surf_data + elev_data
    else:
        # Read data
        if surf_or_elev == 'surf':
            if species == 'dms':
                filename = dms_filename
            else:
                filename = emissions_dir+'p16a_F_'+scenario+'_mam3_'+species+'_surf.nc'
            ds = xr.open_dataset(filename, decode_times=False, drop_variables=['date', ])
        elif surf_or_elev == 'elev':
            filename = emissions_dir+'p16a_F_'+scenario+'_mam3_'+species+'_elev.nc'
            ds = xr.open_dataset(filename, decode_times=False, drop_variables=['date', ])
            # Convert 3D -> 2D
            alt_deltas = (ds['altitude_int'].values[1:] -
                          ds['altitude_int'].values[:-1]) * 100 * 1000  # depth of levels in cm
            alt_deltas = xr.DataArray(alt_deltas, coords={'altitude': ds['altitude'].values},
                                      dims=['altitude', ])  # set coordinates to altitude
            ds = (ds * alt_deltas).sum(dim='altitude').drop('altitude_int')
        else:
            raise ValueError
        # Add time coord; use 2000 as year
        ds['time'] = pd.date_range('2000-01-01', '2000-12-31', freq='MS') + pd.Timedelta('14 days')
        # Calculate annual/seasonal mean, using arithmetic mean across months
        if season == 'annual':
            ds = ds.mean(dim='time')
        else:
            ds = ds.groupby('time.season').mean(dim='time').sel(season=season)
        # Convert to g/m2/yr
        ds = ds * _species_sf_dict[species]
        # Sum across categories
        for var_name in ds.data_vars.keys():
            try:
                ds[species+'_'+surf_or_elev] += ds[var_name]
            except KeyError:
                ds[species+'_'+surf_or_elev] = ds[var_name]
        data = ds[species+'_'+surf_or_elev].load()
    return data


_output_dict = {}  # dictionary to hold data from load_output()


def load_output(variable, scenario='p16a_F_Hist_2000', season='annual', apply_sf=True):
    """
    Load annual/seasonal data for a specific variable and scenario.

    Args:
        variable: string of variable name to load (e.g. 'SWCF_d1', or 'FSNTOA+LWCF')
        scenario: string scenario (default 'p16a_F_Hist_2000')
        season: 'annual' (default) or name of season (e.g 'DJF')
        apply_sf: apply scale factor? (default True)

    Returns:
        xarray DataArray
    """
    # Check if data have been loaded previously
    try:
        data = _output_dict[(variable, scenario, season, apply_sf)]
    except KeyError:
        # Case 1: if '+' in variable name, call recursively
        if '+' in variable:
            variable1, variable2 = variable.split('+')
            data1 = load_output(variable1, scenario=scenario, season=season, apply_sf=apply_sf)
            data2 = load_output(variable2, scenario=scenario, season=season, apply_sf=apply_sf)
            data = data1 + data2
        # Case 2: if '-' in variable name, call recursively
        elif '-' in variable:
            variable1, variable2 = variable.split('-')
            data1 = load_output(variable1, scenario=scenario, season=season, apply_sf=apply_sf)
            data2 = load_output(variable2, scenario=scenario, season=season, apply_sf=apply_sf)
            data = data1 - data2
        # Case 3: variable name is a single variable
        else:
            # Read data
            in_filename = '{}/{}.cam.h0.{}.nc'.format(output_dir, scenario, variable)
            ds = xr.open_dataset(in_filename, decode_times=False)
            # Convert time coordinates
            ds = climapy.cesm_time_from_bnds(ds, min_year=1701)
            # Annual/seasonal mean for each year (Jan-Dec), using arithmetic mean across months
            if season == 'annual':
                data = ds[variable].groupby('time.year').mean(dim='time')
            else:
                data = ds[variable].where(ds['time.season'] == season,
                                          drop=True).groupby('time.year').mean(dim='time')
            # Discard first two years as spin-up
            data = data.where(data['year'] >= 1703, drop=True)
            # Limit time period to max of 60 years
            data = data.where(data['year'] <= 1762, drop=True)
            # Apply scale factor?
            if apply_sf:
                try:
                    data = data * _variable_sf_dict[variable]
                except KeyError:
                    pass
        # Average across level dimension if it exists
        try:
            data = data.mean(dim='lev')
        except ValueError:
            pass
        # Save data for future reference
        _output_dict[(variable, scenario, season, apply_sf)] = data
    return data


_regional_stats_dict = {}  # dictionary to hold results from load_regional_stats()


def load_regional_stats(scenario_combination='All1-All0',
                        variable='SWCF_d1',
                        region='Globe'):
    """
    Load dictionary of regional statistics for a specific variable and scenario.

    Args:
        scenario_combination: scenario combination (default 'All1-All0')
        variable: name of variable (default 'SWCF_d1')
        region: name of region (default 'Globe')

    Returns:
        dictionary, with the following keys:
            scenario_combination: as per input arg
            variable: as per input arg
            region: as per input arg
            awms: area-weighted regional mean annual mean for different years (if single scenario)
            mean: area-weighted regional mean annual mean (averaged across different years)
            error: combined standard error
            ci99: 99% confidence interval (based on 2.576*error)
            p_value: p-value for difference between scenarios
            contributing_scenarios: list of contributing scenarios (e.g. ['All1', 'All0'])
    """
    # Check if regional stats have been calculated previously
    try:
        result = _regional_stats_dict[(scenario_combination, variable, region)]
    except KeyError:
        # Initialise dictionary with input arguments and None
        result = {'scenario_combination': scenario_combination,
                  'variable': variable,
                  'region': region,
                  'awms': None,
                  'mean': None,
                  'error': None,
                  'ci99': None,
                  'p_value': None,
                  'contributing_scenarios': None}
        # Load region bounds
        lon_bounds, lat_bounds = _region_bounds_dict[region]
        # Case 1: scenario_combination is a single scenario
        if scenario_combination in _inverted_scenario_name_dict:
            result['contributing_scenarios'] = [scenario_combination, ]
            # Load annual data
            data = load_output(variable,
                               scenario=_inverted_scenario_name_dict[scenario_combination],
                               season='annual', apply_sf=True)
            # Calculate area-weighted mean for different years
            awms = climapy.xr_area_weighted_stat(data, stat='mean', lon_bounds=lon_bounds,
                                                 lat_bounds=lat_bounds)
            result['awms'] = awms
            # Mean and standard error
            n_years = awms.values.size
            mean = awms.values.mean()  # mean
            error = np.std(awms.values, ddof=1) / np.sqrt(n_years)  # standard error
            result['mean'] = mean
            result['error'] = error
        # Case 2: scenario_combination is a difference between two scenarios
        elif scenario_combination.split('-')[0] in _inverted_scenario_name_dict:
            scenario1, scenario2 = scenario_combination.split('-')
            result['contributing_scenarios'] = [scenario1, scenario2]
            # Call recursively to get stats for each scenario
            stats1 = load_regional_stats(scenario_combination=scenario1,
                                         variable=variable, region=region)
            stats2 = load_regional_stats(scenario_combination=scenario2,
                                         variable=variable, region=region)
            # Combine to get difference between means and the combined error
            mean = stats1['mean'] - stats2['mean']
            error = np.sqrt(stats1['error']**2 + stats2['error']**2)
            result['mean'] = mean
            result['error'] = error
            # p-value based on standard two-sample t-test
            p_value = ttest_ind(stats1['awms'], stats2['awms'], equal_var=True)[1]
            result['p_value'] = p_value
        # Case 3: scenario_combination is ∑(Θ1-All0)
        # Note: ∑(Θ1-All0) = ∑(Θ1)-10xAll0
        elif scenario_combination == '$\Sigma_{\Theta}$($\Theta$1-All0)':
            theta1_scenarios = [s for s in _inverted_scenario_name_dict.keys() if
                                (s[-1] == '1' and s not in ['Correct1', 'All1'])]
            if len(theta1_scenarios) != 10:
                raise RuntimeError('theta1_scenarios = {}'.format(theta1_scenarios))
            result['contributing_scenarios'] = theta1_scenarios + ['All0', ]
            # Call recursively to get mean and error for All0
            temp_stats = load_regional_stats(scenario_combination='All0',
                                             variable=variable, region=region)
            mean_all0 = temp_stats['mean']
            error_all0 = temp_stats['error']
            # Call recursively to get lists of means and errors for each Θ1 scenario
            mean_list = []
            error_list = []
            for scenario in theta1_scenarios:
                temp_stats = load_regional_stats(scenario_combination=scenario,
                                                 variable=variable, region=region)
                mean_list.append(temp_stats['mean'])
                error_list.append(temp_stats['error'])
            # Combine to get sum of means and the combined error
            mean = np.sum(np.array(mean_list)) - 10*mean_all0
            error = np.sqrt((10*error_all0)**2 + np.sum(np.array(error_list)**2))
            result['mean'] = mean
            result['error'] = error
        # Case 4: scenario_combination is ∑(All1-Θ0)
        # Note: ∑(All1-Θ0) = 10xAll1-∑(Θ0)
        elif scenario_combination == '$\Sigma_{\Theta}$(All1-$\Theta$0)':
            theta0_scenarios = [s for s in _inverted_scenario_name_dict.keys() if
                                (s[-1] == '0' and s != 'All0')]
            if len(theta0_scenarios) != 10:
                raise RuntimeError('theta0_scenarios = {}'.format(theta0_scenarios))
            result['contributing_scenarios'] = theta0_scenarios + ['All1', ]
            # Call recursively to get mean and error for All1
            temp_stats = load_regional_stats(scenario_combination='All1',
                                             variable=variable, region=region)
            mean_all1 = temp_stats['mean']
            error_all1 = temp_stats['error']
            # Call recursively to get lists of means and errors for each Θ0 scenario
            mean_list = []
            error_list = []
            for scenario in theta0_scenarios:
                temp_stats = load_regional_stats(scenario_combination=scenario,
                                                 variable=variable, region=region)
                mean_list.append(temp_stats['mean'])
                error_list.append(temp_stats['error'])
            # Combine to get sum of means and the combined error
            mean = 10*mean_all1 - np.sum(np.array(mean_list))
            error = np.sqrt((10*error_all1)**2 + np.sum(np.array(error_list)**2))
            result['mean'] = mean
            result['error'] = error
        else:
            raise ValueError('scenario_combination not recognized')
        # 99% confidence interval based on standard error
        ci99 = (mean - 2.576 * error, mean + 2.576 * error)
        result['ci99'] = ci99
        # Save result for future reference
        _regional_stats_dict[(scenario_combination, variable, region)] = result
    # Return result
    return result


_zonal_stats_dict = {}  # dictionary to hold results from load_zonal_stats()


def load_zonal_stats(scenario_combination='All1-All0',
                     variable='SWCF_d1',
                     lon_bounds=None):
    """
    Load dictionary of zonal mean statistics for a specific variable and scenario.

    Args:
        scenario_combination: scenario combination (default 'All1-All0')
        variable: name of variable (default 'SWCF_d1')
        lon_bounds: tuple containing longitude bounds (default None)

    Returns:
        dictionary, with the following keys:
            scenario_combination: as per input arg
            variable: as per input arg
            lon_bounds: as per input arg
            zonmeans: DataArray of zonal-mean annual-means for different years (if single scenario)
            mean: DataArray of zonal-mean annual-mean (averaged across different years)
            error: DataArray of combined standard errors
            ci99: DataArray of 99% confidence intervals (based on 2.576*error)
            p_value: array of p-values for difference between scenarios
            contributing_scenarios: list of contributing scenarios (e.g. ['All1', 'All0'])
    """
    # Check if zonal stats have been calculated previously
    try:
        result = _zonal_stats_dict[(scenario_combination, variable, lon_bounds)]
    except KeyError:
        # Initialise dictionary with input arguments and None
        result = {'scenario_combination': scenario_combination,
                  'variable': variable,
                  'lon_bounds': lon_bounds,
                  'zonmeans': None,
                  'mean': None,
                  'error': None,
                  'ci99': None,
                  'p_value': None,
                  'contributing_scenarios': None}
        # Case 1: scenario_combination is a single scenario
        if scenario_combination in _inverted_scenario_name_dict:
            result['contributing_scenarios'] = [scenario_combination, ]
            # Load annual data
            data = load_output(variable,
                               scenario=_inverted_scenario_name_dict[scenario_combination],
                               season='annual', apply_sf=True)
            # Limit to specific longitude bounds?
            if lon_bounds is not None:
                data = climapy.xr_mask_bounds(data, lon_bounds=lon_bounds, lat_bounds=None)
                data = data.dropna(dim='lon', how='all')
            # Calculate zonal means for different years
            zonmeans = data.mean(dim='lon')
            result['zonmeans'] = zonmeans
            # Mean and standard error across years
            n_years = zonmeans['year'].size
            mean = zonmeans.mean(dim='year')  # mean across years
            error = zonmeans.std(dim='year', ddof=1) / np.sqrt(n_years)  # standard error
            result['mean'] = mean
            result['error'] = error
        # Case 2: scenario_combination is a difference between two scenarios
        elif scenario_combination.split('-')[0] in _inverted_scenario_name_dict:
            scenario1, scenario2 = scenario_combination.split('-')
            result['contributing_scenarios'] = [scenario1, scenario2]
            # Call recursively to get zonal stats for each scenario
            stats1 = load_zonal_stats(scenario_combination=scenario1, variable=variable,
                                      lon_bounds=lon_bounds)
            stats2 = load_zonal_stats(scenario_combination=scenario2, variable=variable,
                                      lon_bounds=lon_bounds)
            # Combine to get difference between means and the combined error
            mean = stats1['mean'] - stats2['mean']
            error = np.sqrt(stats1['error']**2 + stats2['error']**2)
            result['mean'] = mean
            result['error'] = error
            # p-value based on standard two-sample t-test
            p_value = ttest_ind(stats1['zonmeans'], stats2['zonmeans'], equal_var=True)[1]
            result['p_value'] = p_value
        # Case 3: scenario_combination is ∑(Θ1-All0)
        # Note: ∑(Θ1-All0) = ∑(Θ1)-10xAll0
        elif scenario_combination == '$\Sigma_{\Theta}$($\Theta$1-All0)':
            theta1_scenarios = [s for s in _inverted_scenario_name_dict.keys() if
                                (s[-1] == '1' and s not in ['Correct1', 'All1'])]
            if len(theta1_scenarios) != 10:
                raise RuntimeError('theta1_scenarios = {}'.format(theta1_scenarios))
            result['contributing_scenarios'] = theta1_scenarios + ['All0', ]
            # Call recursively to get mean and error for All0
            temp_stats = load_zonal_stats(scenario_combination='All0',
                                          variable=variable, lon_bounds=lon_bounds)
            mean_all0 = temp_stats['mean']
            error_all0 = temp_stats['error']
            # Call recursively to get lists of means and errors for each Θ1 scenario
            mean_list = []
            error_list = []
            for scenario in theta1_scenarios:
                temp_stats = load_zonal_stats(scenario_combination=scenario,
                                              variable=variable,
                                              lon_bounds=lon_bounds)
                mean_list.append(temp_stats['mean'])
                error_list.append(temp_stats['error'])
            # Combine to get sum of means and the combined error
            mean = sum(mean_list) - 10*mean_all0
            error = np.sqrt((10*error_all0)**2 + sum([e**2 for e in error_list]))
            result['mean'] = mean
            result['error'] = error
        # Case 4: scenario_combination is ∑(All1-Θ0)
        elif scenario_combination == '$\Sigma_{\Theta}$(All1-$\Theta$0)':
            theta0_scenarios = [s for s in _inverted_scenario_name_dict.keys() if
                                (s[-1] == '0' and s != 'All0')]
            if len(theta0_scenarios) != 10:
                raise RuntimeError('theta0_scenarios = {}'.format(theta0_scenarios))
            result['contributing_scenarios'] = theta0_scenarios + ['All1', ]
            # Call recursively to get mean and error for All1
            temp_stats = load_zonal_stats(scenario_combination='All1',
                                          variable=variable, lon_bounds=lon_bounds)
            mean_all1 = temp_stats['mean']
            error_all1 = temp_stats['error']
            # Call recursively to get lists of means and errors for each Θ0 scenario
            mean_list = []
            error_list = []
            for scenario in theta0_scenarios:
                temp_stats = load_zonal_stats(scenario_combination=scenario,
                                              variable=variable,
                                              lon_bounds=lon_bounds)
                mean_list.append(temp_stats['mean'])
                error_list.append(temp_stats['error'])
            # Combine to get sum of means and the combined error
            mean = 10*mean_all1 - sum(mean_list)
            error = np.sqrt((10*error_all1)**2 + sum([e ** 2 for e in error_list]))
            result['mean'] = mean
            result['error'] = error
        # Case 5: scenario_combination is a fraction between two scenarios
        elif scenario_combination.split('/')[0] in _inverted_scenario_name_dict:
            scenario1, scenario2 = scenario_combination.split('/')
            result['contributing_scenarios'] = [scenario1, scenario2]
            # Call recursively to get zonal stats for each scenario
            stats1 = load_zonal_stats(scenario_combination=scenario1, variable=variable,
                                      lon_bounds=lon_bounds)
            stats2 = load_zonal_stats(scenario_combination=scenario2, variable=variable,
                                      lon_bounds=lon_bounds)
            # Combine to get fraction and combined error
            mean = stats1['mean'] / stats2['mean']
            error = np.abs(mean) * np.sqrt((stats1['error'] / stats1['mean']) ** 2 +
                                           (stats2['error'] / stats2['mean']) ** 2)
            result['mean'] = mean
            result['error'] = error
            # p-value based on standard two-sample t-test
            p_value = ttest_ind(stats1['zonmeans'], stats2['zonmeans'], equal_var=True)[1]
            result['p_value'] = p_value
        else:
            raise ValueError('scenario_combination not recognized')
        # 99% confidence interval based on standard error
        ci99 = (mean - 2.576 * error, mean + 2.576 * error)
        result['ci99'] = ci99
        # Save result for future reference
        _zonal_stats_dict[(scenario_combination, variable, lon_bounds)] = result
    # Return result
    return result


_2d_stats_dict = {}  # dictionary to hold results from load_2d_stats()


def load_2d_stats(scenario_combination='All1-All0',
                  variable='SWCF_d1'):
    """
    Load dictionary of 2D statistics, for all lons and lats, for a specific variable and scenario.

    Args:
        scenario_combination: scenario combination (default 'All1-All0')
        variable: name of variable (default 'SWCF_d1')

    Returns:
        dictionary, with the following keys:
            scenario_combination: as per input arg
            variable: as per input arg
            data: DataArray of annual-means for different years (if single scenario)
            mean: DataArray of annual-mean averaged across different years
            error: DataArray of combined standard errors
            ci99: tuple of DataArrays of 99% confidence intervals (based on 2.576*error)
            p_value: array of p-values for difference between scenarios
            contributing_scenarios: list of contributing scenarios (e.g. ['All1', 'All0'])
    """
    # Check if 2D stats have been calculated previously
    try:
        result = _2d_stats_dict[(scenario_combination, variable)]
    except KeyError:
        # Initialise dictionary with input arguments and None
        result = {'scenario_combination': scenario_combination,
                  'variable': variable,
                  'data': None,
                  'mean': None,
                  'error': None,
                  'ci99': None,
                  'p_value': None,
                  'contributing_scenarios': None}
        # Case 1: scenario_combination is a single scenario
        if scenario_combination in _inverted_scenario_name_dict:
            result['contributing_scenarios'] = [scenario_combination, ]
            # Load annual data
            data = load_output(variable,
                               scenario=_inverted_scenario_name_dict[scenario_combination],
                               season='annual', apply_sf=True)
            data = climapy.xr_shift_lon(data, lon_min=-179.)  # shift longitudes
            result['data'] = data
            # Mean and standard error across years
            n_years = data['year'].size
            mean = data.mean(dim='year')  # mean across years
            error = data.std(dim='year', ddof=1) / np.sqrt(n_years)  # standard error
            result['mean'] = mean
            result['error'] = error
        # Case 2: scenario_combination is a difference between two scenarios
        elif (len(scenario_combination.split('-')) == 2 and
              scenario_combination.split('-')[0] in _inverted_scenario_name_dict):
            scenario1, scenario2 = scenario_combination.split('-')
            result['contributing_scenarios'] = [scenario1, scenario2]
            # Call recursively to get 2D stats for each scenario
            stats1 = load_2d_stats(scenario_combination=scenario1, variable=variable)
            stats2 = load_2d_stats(scenario_combination=scenario2, variable=variable)
            # Combine to get difference between means and the combined error
            mean = stats1['mean'] - stats2['mean']
            error = np.sqrt(stats1['error']**2 + stats2['error']**2)
            result['mean'] = mean
            result['error'] = error
            # p-value based on standard two-sample t-test
            p_value = ttest_ind(stats1['data'], stats2['data'], equal_var=True)[1]
            result['p_value'] = p_value
        # Case 3: scenario_combination is ∑(Θ1-All0)
        # Note: ∑(Θ1-All0) = ∑(Θ1)-10xAll0
        elif scenario_combination == '$\Sigma_{\Theta}$($\Theta$1-All0)':
            theta1_scenarios = [s for s in _inverted_scenario_name_dict.keys() if
                                (s[-1] == '1' and s not in ['Correct1', 'All1'])]
            if len(theta1_scenarios) != 10:
                raise RuntimeError('theta1_scenarios = {}'.format(theta1_scenarios))
            result['contributing_scenarios'] = theta1_scenarios + ['All0', ]
            # Call recursively to get mean and error for All0
            temp_stats = load_2d_stats(scenario_combination='All0', variable=variable)
            mean_all0 = temp_stats['mean']
            error_all0 = temp_stats['error']
            # Call recursively to get lists of means and errors for each Θ1 scenario
            mean_list = []
            error_list = []
            for scenario in theta1_scenarios:
                temp_stats = load_2d_stats(scenario_combination=scenario,
                                           variable=variable)
                mean_list.append(temp_stats['mean'])
                error_list.append(temp_stats['error'])
            # Combine to get sum of means and the combined error
            mean = sum(mean_list) - 10*mean_all0
            error = np.sqrt((10*error_all0)**2 + sum([e**2 for e in error_list]))
            result['mean'] = mean
            result['error'] = error
        # Case 4: scenario_combination is ∑(All1-Θ0)
        elif scenario_combination == '$\Sigma_{\Theta}$(All1-$\Theta$0)':
            theta0_scenarios = [s for s in _inverted_scenario_name_dict.keys() if
                                (s[-1] == '0' and s != 'All0')]
            if len(theta0_scenarios) != 10:
                raise RuntimeError('theta0_scenarios = {}'.format(theta0_scenarios))
            result['contributing_scenarios'] = theta0_scenarios + ['All1', ]
            # Call recursively to get mean and error for All1
            temp_stats = load_2d_stats(scenario_combination='All1', variable=variable)
            mean_all1 = temp_stats['mean']
            error_all1 = temp_stats['error']
            # Call recursively to get lists of means and errors for each Θ0 scenario
            mean_list = []
            error_list = []
            for scenario in theta0_scenarios:
                temp_stats = load_2d_stats(scenario_combination=scenario,
                                           variable=variable)
                mean_list.append(temp_stats['mean'])
                error_list.append(temp_stats['error'])
            # Combine to get sum of means and the combined error
            mean = 10*mean_all1 - sum(mean_list)
            error = np.sqrt((10*error_all1)**2 + sum([e ** 2 for e in error_list]))
            result['mean'] = mean
            result['error'] = error
        # Case 5: scenario_combination is ∑(All1-Θ0)-∑(Θ1-All0)
        elif scenario_combination == ('$\Sigma_{\Theta}$(All1-$\Theta$0)-'
                                      '$\Sigma_{\Theta}$($\Theta$1-All0)'):
            scenario_combination1 = '$\Sigma_{\Theta}$(All1-$\Theta$0)'
            scenario_combination2 = '$\Sigma_{\Theta}$($\Theta$1-All0)'
            # Call recursively to get 2D stats for each scenario combination
            stats1 = load_2d_stats(scenario_combination=scenario_combination1, variable=variable)
            stats2 = load_2d_stats(scenario_combination=scenario_combination2, variable=variable)
            # Contributing scenarios
            result['contributing_scenarios'] = (stats1['contributing_scenarios'] +
                                                stats2['contributing_scenarios'])
            # Combine to get difference between means and the combined error
            mean = stats1['mean'] - stats2['mean']
            error = np.sqrt(stats1['error']**2 + stats2['error']**2)
            result['mean'] = mean
            result['error'] = error
        # Case 6: scenario_combination is mean of two differences
        #     e.g. '((EAs1-All0)+(All1-EAs0))/2'
        elif len(scenario_combination.split('+')) == 2 and scenario_combination[-2:] == '/2':
            scenario_combination1, scenario_combination2 = \
                scenario_combination[:-2].replace('(', '').replace(')', '').split('+')
            # Call recursively to get 2D stats for each scenario combination
            stats1 = load_2d_stats(scenario_combination=scenario_combination1, variable=variable)
            stats2 = load_2d_stats(scenario_combination=scenario_combination2, variable=variable)
            # Contributing scenarios
            result['contributing_scenarios'] = (stats1['contributing_scenarios'] +
                                                stats2['contributing_scenarios'])
            # Combine to get means of means and the combined error
            mean = (stats1['mean'] + stats2['mean']) / 2
            error = (np.sqrt(stats1['error'] ** 2 + stats2['error'] ** 2)) / 2
            result['mean'] = mean
            result['error'] = error
        else:
            raise ValueError('scenario_combination not recognized')
        # 99% confidence interval based on standard error
        ci99 = (mean - 2.576 * error, mean + 2.576 * error)
        result['ci99'] = ci99
        # Save result for future reference
        _2d_stats_dict[(scenario_combination, variable)] = result
    # Return result
    return result


def load_landfrac():
    """
    Load land fraction (LANDFRAC).
    LANDFRAC is invariant across time and scenarios.

    Returns:
        xarray DataArray
    """
    # Read data
    in_filename = '{}/p16a_F_Hist_2000.cam.h0.LANDFRAC.nc'.format(output_dir)
    ds = xr.open_dataset(in_filename, decode_times=False)
    # Convert time coordinates
    ds = climapy.cesm_time_from_bnds(ds, min_year=1701)
    # Collapse time dimension (by calculating mean across time)
    data = ds['LANDFRAC'].mean(dim='time')
    return data
