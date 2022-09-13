import numpy as np
import pandas as pd

import cri_calc as cri
import cri_helper as helper


def calculate_cri(rec_own, rec_target):
    own = rec_own._asdict()
    target = rec_target._asdict()
    
    ves_dcpa, ves_tcpa, hr, rel_movement_angle, dist_euclid, speed_r = cri.colregs_alarms(own=own, target=target)
    ves_cri = cri.calculate_cri(own, target, ves_dcpa, ves_tcpa, hr, rel_movement_angle, dist_euclid, speed_r)
    
    return ves_dcpa, ves_tcpa, hr, rel_movement_angle, dist_euclid, speed_r, [ves_cri]


def calc_cri_ours_with_own_length(rec_own, rec_target):
    own = rec_own._asdict()
    target = rec_target._asdict()
    
    own_geom_nm, target_geom_nm = map(helper.angular_to_nautical_miles, [own['geom'], target['geom']])
    xr, yr = helper.calculate_delta(own_geom_nm.x, target_geom_nm.x), helper.calculate_delta(own_geom_nm.y, target_geom_nm.y)
    hr = helper.calculate_delta(own['course'], target['course'])
    
    # Get vessels' Euclidean Distance -- NAUTICAL MILES
    dist_euclid = np.sqrt(xr**2 + yr**2)
    
    return dist_euclid, [dist_euclid, own['speed'], target['speed'], own['course'], target['course'], own['length']]


def calc_cri_ours_with_target_length(rec_own, rec_target):
    own = rec_own._asdict()
    target = rec_target._asdict()
    
    own_geom_nm, target_geom_nm = map(helper.angular_to_nautical_miles, [own['geom'], target['geom']])
    xr, yr = helper.calculate_delta(own_geom_nm.x, target_geom_nm.x), helper.calculate_delta(own_geom_nm.y, target_geom_nm.y)
    hr = helper.calculate_delta(own['course'], target['course'])
    
    # Get vessels' Euclidean Distance -- NAUTICAL MILES
    dist_euclid = np.sqrt(xr**2 + yr**2)
    
    return dist_euclid, [dist_euclid, own['speed'], target['speed'], own['course'], target['course'], target['length']]


def calc_cri_ours_with_both_length(rec_own, rec_target):
    own = rec_own._asdict()
    target = rec_target._asdict()
    
    own_geom_nm, target_geom_nm = map(helper.angular_to_nautical_miles, [own['geom'], target['geom']])
    xr, yr = helper.calculate_delta(own_geom_nm.x, target_geom_nm.x), helper.calculate_delta(own_geom_nm.y, target_geom_nm.y)
    hr = helper.calculate_delta(own['course'], target['course'])
    
    # Get vessels' Euclidean Distance -- NAUTICAL MILES
    dist_euclid = np.sqrt(xr**2 + yr**2)
    
    return dist_euclid, [dist_euclid, own['speed'], target['speed'], own['course'], target['course'], own['length'], target['length']]


def calc_cri_ours_with_no_length(rec_own, rec_target):
    own = rec_own._asdict()
    target = rec_target._asdict()
    
    own_geom_nm, target_geom_nm = map(helper.angular_to_nautical_miles, [own['geom'], target['geom']])
    xr, yr = helper.calculate_delta(own_geom_nm.x, target_geom_nm.x), helper.calculate_delta(own_geom_nm.y, target_geom_nm.y)
    hr = helper.calculate_delta(own['course'], target['course'])
    
    # Get vessels' Euclidean Distance -- NAUTICAL MILES
    dist_euclid = np.sqrt(xr**2 + yr**2)
    
    return dist_euclid, [dist_euclid, own['speed'], target['speed'], own['course'], target['course']]


def calc_cri_ours(rec_own, rec_target):
    own = rec_own._asdict()
    target = rec_target._asdict()
    
    own_geom_nm, target_geom_nm = map(helper.angular_to_nautical_miles, [own['geom'], target['geom']])
    xr, yr = helper.calculate_delta(own_geom_nm.x, target_geom_nm.x), helper.calculate_delta(own_geom_nm.y, target_geom_nm.y)
    hr = helper.calculate_delta(own['course'], target['course'])
    
    # Get vessels' Euclidean Distance -- NAUTICAL MILES
    dist_euclid = np.sqrt(xr**2 + yr**2)
    
    return dist_euclid, [dist_euclid, own['speed'], target['speed'], own['course'], target['course'], own['length'], target['length']]


def calc_cri_park_etal(rec_own, rec_target):
    own = rec_own._asdict()
    target = rec_target._asdict()
    
    own_geom_nm, target_geom_nm = map(helper.angular_to_nautical_miles, [own['geom'], target['geom']])
    xr, yr = helper.calculate_delta(own_geom_nm.x, target_geom_nm.x), helper.calculate_delta(own_geom_nm.y, target_geom_nm.y)
    hr = helper.calculate_delta(own['course'], target['course'])
    
    # Get vessels' Euclidean Distance -- NAUTICAL MILES
    dist_euclid = np.sqrt(xr**2 + yr**2)
    
    return dist_euclid, [dist_euclid, hr, own['speed'], target['speed'], own['course'], target['course'], own['length'], target['length']]


def calc_cri_li_etal(rec_own, rec_target):
    own = rec_own._asdict()
    target = rec_target._asdict()
    
    ves_dcpa, ves_tcpa, hr, rel_movement_angle, dist_euclid, speed_r = cri.colregs_alarms(own=own, target=target)
    
    return dist_euclid, [speed_r, hr, rel_movement_angle, dist_euclid]


def calc_cri_gang_etal(rec_own, rec_target):
    own = rec_own._asdict()
    target = rec_target._asdict()
    
    own_geom_nm, target_geom_nm = map(helper.angular_to_nautical_miles, [own['geom'], target['geom']])
    xr, yr = helper.calculate_delta(own_geom_nm.x, target_geom_nm.x), helper.calculate_delta(own_geom_nm.y, target_geom_nm.y)
    hr = helper.calculate_delta(own['course'], target['course'])
    
    # Get vessels' Euclidean Distance -- NAUTICAL MILES
    dist_euclid = np.sqrt(xr**2 + yr**2)
    
    return dist_euclid, [own['course'], target['course'], own['speed'], target['speed'], hr, dist_euclid]


def calc_cri_timeslice(df, **kwargs):
    timeslice_result = []
    
    for row_i in df.itertuples():
        for row_j in df.itertuples():
            if row_i.Index == row_j.Index:
                continue
                
            timeslice_result.append([row_i.Index, row_i.mmsi, row_i.geom, row_i.speed, row_i.course, 
                                     row_j.Index, row_j.mmsi, row_j.geom, row_j.speed, row_j.course, *calc_cri(row_i, row_j, **kwargs)])
            
#     return pd.DataFrame(timeslice_result, columns=['own', 'target', 'dcpa', 'tcpa', 'hr', 'rel_movement_angle', 'dist_euclid', 'speed_r', 'cri'])
    return pd.DataFrame(timeslice_result, columns=['own_Index', 'own_mmsi', 'own_geom', 'own_speed', 'own_course',
                                                   'target_Index', 'target_mmsi', 'target_geom', 'target_speed', 'target_course', 
                                                   'dist_euclid', 'cri'])


def calc_cri(rec_own, rec_target, model=None, model_fun=calculate_cri, model_norm=None):
    own = rec_own
    target = rec_target
    
    if model is None:
        _, _, _, _, dist_euclid, _, ves_cri = model_fun(own, target)
    else:
        dist_euclid, model_input = model_fun(own, target)
        ves_cri = model.predict(model_norm.transform(np.array(model_input).reshape(1, -1)))
    
    return dist_euclid, min(max(ves_cri[0], 0), 1)