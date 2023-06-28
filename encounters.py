import numpy as np
import pandas as pd
import shapely as shp

import itertools

import tqdm
from sklearn.neighbors import KDTree, BallTree
from joblib import Parallel, delayed

import cri_calc as cri
import cri_helper as helper

EPS = 1e-9


def calculate_cri(rec_own, rec_target, stationary_speed_threshold=1, vcra_model=None):
    own = rec_own.to_dict()
    target = rec_target.to_dict()

    # First things first, convert input features to proper UoM
    # ## Coordinates: from ANGULAR to NAUTICAL MILES
    # ## Bearing: from DEGREES to RADIANS
    # ## Speed: from XXXX to KNOTS (in most AIS datasets it's KNOTS, so it's OK-ish)
    # ## Lengths: from METERS to NAUTICAL MILES
    own['course_rad'], target['course_rad'] = helper.degrees_to_radians(own['course']), \
                                              helper.degrees_to_radians(target['course'])

    own['length_nmi'], target['length_nmi'] = own['length'] / 1852, target['length'] / 1852

    # ###
    dist_euclid, speed_rx, speed_ry, speed_r, rel_movement_direction, azimuth_angle_target_to_own, \
    relative_bearing_target_to_own, DCPA, TCPA = cri.calculate_cpa(own=own, target=target)

    if vcra_model is None:
        U_dcpa, U_tcpa, U_dist, U_bearing, U_speed, ves_cri = cri.calculate_cri(
            own, target, dist_euclid, speed_r, rel_movement_direction,
            azimuth_angle_target_to_own, relative_bearing_target_to_own, DCPA, TCPA, 
            stationary_speed_threshold=stationary_speed_threshold
        )
    else:
         U_dcpa, U_tcpa, U_dist, U_bearing, U_speed, ves_cri = cri.calculate_cri_vcra(
            own, target, dist_euclid, rel_movement_direction, azimuth_angle_target_to_own, TCPA, 
            vcra_model, stationary_speed_threshold=1
        )
        
    return dist_euclid, speed_rx, speed_ry, speed_r, rel_movement_direction, azimuth_angle_target_to_own, \
           relative_bearing_target_to_own, DCPA, TCPA, U_dcpa, U_tcpa, U_dist, U_bearing, U_speed, ves_cri
    
   
    

def pairs_in_radius(df, diam=1000):
    '''
        Get all pairs with distance <= diam
    '''

    # 25/07/2020 - Calculating (Euclidean) Distance using KD-Trees (fast)
    # Note: sklearn.neighbors **does** support Haversine Metric
    # -------------------------------------------------------------------
    df_rad = np.deg2rad(df[df.columns[::-1]])
    df_kdtree = BallTree(df_rad, leaf_size=40,
                         metric='haversine')
    # The I/O in sklearn's haversine method is in radians
    # (22/01/2023) and coordinates in reverse order, apparently...

    # SciPy and ScikitLearn's Implementation of the BallTree Spatial Index is less-than-or-equal to a specified radius.
    theta = diam / (6371 * 1000)  # radians = distance (unit) / earth_radius (unit)
    point_neighbors, dist = df_kdtree.query_radius(df_rad, return_distance=True, sort_results=True, r=theta)

    mask = np.vectorize(lambda l: len(l) > 1)(point_neighbors)
    return point_neighbors[mask], dist[mask] * 6371 * 1000


def translate(timeslice, index, oid_name='mmsi'):
    return timeslice.iloc[index][oid_name]


def get_nearest_neighbors(timeslice, coords=['lon', 'lat'], diam=1852):
    # (2023/05/06) Due to AIS/GPS inacuraccy, vessels' coordinates may overlap, leading to consistency issues. 
    # Shifting the coordinates EPS 1 micron towards NE may help resolve this issue...
    timeslice_coords = timeslice[coords].copy()
    timeslice_coords.loc[timeslice_coords.duplicated()] += EPS

    neighbors, distance = pairs_in_radius(timeslice_coords, diam=diam)
    return neighbors, distance


def get_pairs(pairs, distances):
    return list(zip(itertools.repeat(pairs[0]), pairs[1:], distances[1:]))


def get_kinematics(vessel):
    return [vessel.name, vessel.mmsi, vessel.geometry, vessel.speed, vessel.course, vessel.length]


def get_key(timeslice, vessel_i, vessel_j, **kwargs):
    vessel_id_i, vessel_id_j = translate(timeslice, vessel_i, **kwargs), translate(timeslice, vessel_j, **kwargs)
    return tuple([vessel_id_i, vessel_id_j])


def current_pairs(dt, curr, coords=['lon', 'lat'], diam=1852, stationary_speed_threshold=1, **kwargs):
    if curr.empty:
        print('\t\t----- Timeslice is EMPTY; No Pairs Discovered! -----')
        return pd.DataFrame(
            data=[], 
            columns=['pair', 'dist', 'start', 'end', 'geometry', 'kinematics_own', 'kinematics_target']
        ).set_index('pair')
        
    pairs, dists = get_nearest_neighbors(curr, coords, diam)

    coords_loc = [curr.columns.get_loc(coord) for coord in coords]
    curr_pairs = itertools.chain.from_iterable([get_pairs(pairs, dists) for pairs, dists in zip(pairs, dists)])

    curr_pairs_mi = pd.DataFrame(
        data=[[
            get_key(curr, v_i, v_j, **kwargs), dist, dt, dt, [curr.iloc[v_i, coords_loc]],
            curr.iloc[v_i], curr.iloc[v_j]
        ] for dt, (v_i, v_j, dist) in zip(itertools.repeat(dt), curr_pairs) if curr.iloc[v_i].speed > stationary_speed_threshold],
        columns=['pair', 'dist', 'start', 'end', 'geometry', 'kinematics_own', 'kinematics_target']
    ).set_index('pair')

    assert not curr_pairs_mi.index.duplicated().any(), ValueError('[current_pairs] Violation of Primary Key Constraint...')
    return curr_pairs_mi

def query_nans(curr_pairs, feature, idx_slice):
    mask_nan = curr_pairs[feature].isna()
    nan_pairs = curr_pairs.loc[mask_nan, idx_slice].copy()

    # ## Amend column names; Remove "_prev" suffix from columns (in case they exist)
    nan_pairs.columns = nan_pairs.columns.str.rstrip('_prev')
    return mask_nan, nan_pairs


def update_inactives(inactive_pairs, candidate_pairs, dt_thresh):
    mask_duration = (candidate_pairs.end - candidate_pairs.start) >= dt_thresh
    return pd.concat((inactive_pairs, candidate_pairs.loc[mask_duration]), ignore_index=False)


def curr_encounters(curr_pairs, mask_emerged, mask_inactive, active_idx_slice=pd.IndexSlice['dist':],
                    inactive_idx_slice=pd.IndexSlice[:'kinematics_target_prev']):
    alive_pairs = curr_pairs.loc[(~mask_emerged) & (~mask_inactive)].copy()

    mask_distance = alive_pairs.dist_prev >= alive_pairs.dist  # ## Find monotonically decreasing (w.r.t distance) pairs

    inactive_alive_pairs = alive_pairs.loc[~mask_distance, inactive_idx_slice].copy()

    # ## Amend column names; Remove "_prev" suffix from columns
    inactive_alive_pairs.columns = inactive_alive_pairs.columns.str.rstrip('_prev')

    # print(f'{alive_pairs=}')
    # print(f'{mask_distance=}')
    # print(f'{~mask_distance=}')
    # print(f'{alive_pairs.index[~mask_distance]=}')
    # print(f'{alive_pairs.drop(alive_pairs.index[~mask_distance], axis=0)=}')

    # ## Get the pairs which still satisfy distance monotonicity; Update their starting Timestamps and LineStrings
    active_alive_pairs = alive_pairs.drop(alive_pairs.index[~mask_distance], axis=0)
    active_alive_pairs.start = active_alive_pairs.start_prev
    active_alive_pairs.geometry = active_alive_pairs.geometry_prev + active_alive_pairs.geometry

    return active_alive_pairs.loc[:, active_idx_slice], inactive_alive_pairs


def encountering_vessels_timeslice(curr_pairs, inactive_pairs, dt_thresh,
                                   active_idx_slice=pd.IndexSlice['dist':],
                                   inactive_idx_slice=pd.IndexSlice[:'kinematics_target_prev']):
    '''
        TODO: Docstring
    '''
    # Case #2 - ```curr_pairs''' not in ```active_pairs```; Add to ```active_pairs```
    mask_emerged, emerged_pairs = query_nans(curr_pairs, 'dist_prev', active_idx_slice)

    # Case #3 - ```active_pairs``` not in ```curr_pairs```; Add to ```inactive_pairs```
    mask_inactive, disappeared_pairs = query_nans(curr_pairs, 'dist', inactive_idx_slice)

    # If ** temporal constraints ** are satisfied --> Add to ```inactive_pairs```
    inactive_pairs = update_inactives(inactive_pairs, disappeared_pairs, dt_thresh)

    # Case #4 - ```curr_pairs''' in ```active_pairs```; Compare ```prev``` vs. ```curr``` distances
    active_alive_pairs, inactive_alive_pairs = curr_encounters(
        curr_pairs, mask_emerged, mask_inactive, active_idx_slice, inactive_idx_slice
    )

    # Add to ```inactive_pairs``` the ```inactive_alive_pairs``` which stopped satisfying
    # ** distance monotonicity **, yet they satisfy ** temporal constraints **
    inactive_pairs = update_inactives(inactive_pairs, inactive_alive_pairs, dt_thresh)

    # Finally - Refresh ```active_pairs``` DataFrame
    active_pairs = pd.concat((emerged_pairs, active_alive_pairs), ignore_index=False)

    return active_pairs, inactive_pairs


def encountering_vessels(data, oid_name='mmsi', dt_name='datetime', dt_thresh=pd.Timedelta(1, unit="min"),
                         coords=['lon', 'lat'], diam=1852, stationary_speed_threshold=1, vcra_model=None, **kwargs):
    CRI_DATASET_FEATURES = [
        'own_index', 'own_mmsi', 'own_geometry', 'own_speed', 'own_course', 'own_length',
        'target_index', 'target_mmsi', 'target_geometry', 'target_speed', 'target_course', 'target_length',
        'dist_euclid', 'speed_rx', 'speed_ry', 'speed_r', 'rel_movement_direction', 'azimuth_angle_target_to_own',
        'relative_bearing_target_to_own',
        'dcpa', 'tcpa', 'U_dcpa', 'U_tcpa', 'U_dist', 'U_bearing', 'U_speed', 'ves_cri'
    ]  # 27 Features

    prior_pairs = kwargs.pop('prior', None)
    (active_pairs, inactive_pairs), results = prior_pairs if prior_pairs is not None else (pd.DataFrame(), pd.DataFrame()), []
    print(f'Prior: {len(active_pairs)=}, {len(inactive_pairs)=}, {len(results)=}')

    for dt, curr in tqdm.tqdm(data.groupby(dt_name)):
        curr_slice = current_pairs(dt, curr, oid_name=oid_name, coords=coords, diam=diam, stationary_speed_threshold=stationary_speed_threshold)

        # Case #1 - ```active_pairs``` is empty; All Current Pairs are encountering "candidates"
        if active_pairs.empty:
            active_pairs = curr_slice.copy()
            continue

        ''' 
            Get the Current view of previous vs. current pairs; Columns: 
            < dist_prev, start_prev, end_prev, kinematics_own_prev, kinematics_target_prev, 
              dist, start, end, kinematics_own, kinematics_target> 
        '''
        curr_pairs = pd.merge(
            active_pairs, curr_slice, how='outer', left_index=True, right_index=True, suffixes=('_prev', '')
        )

        active_pairs, inactive_pairs = encountering_vessels_timeslice(curr_pairs, inactive_pairs, dt_thresh)

        '''    
            For the ```active_pairs``` which satisfy ** temporal constraints **, calculate the vessels' CRI
        '''
        # ## Calculate vessels' CRI
        active_pairs_cri = active_pairs.loc[active_pairs.end - active_pairs.start >= dt_thresh].apply(
            lambda l: [
                *get_kinematics(l.kinematics_own), *get_kinematics(l.kinematics_target),
                *calculate_cri(l.kinematics_own, l.kinematics_target, stationary_speed_threshold=stationary_speed_threshold, vcra_model=vcra_model)
            ], axis=1
        ).values.tolist()

        results.append(pd.concat(
            {dt: pd.DataFrame(active_pairs_cri, columns=CRI_DATASET_FEATURES)}, names=[dt_name, ]
        ))

    merge_output = kwargs.pop('merge_output', True)

    if merge_output:
        return pd.concat((active_pairs, inactive_pairs)), pd.concat(results)
    
    return active_pairs, inactive_pairs, pd.concat(results)
