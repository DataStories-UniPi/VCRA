import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, shape

import rdp

try:
    import osmnx as ox
except ModuleNotFoundError:
    ox = None

import similaritymeasures
from scipy.interpolate import interp1d
from sklearn.cluster import OPTICS

import re
import tqdm
import multiprocessing
from joblib import delayed, Parallel


# Earth's radius in kilometers
EARTH_RADIUS = 6371


# Get the coordinates of a Shapely Geometry (e.g. Point, Polygon, etc.) as NumPy array
shapely_coords_numpy = lambda l: np.array(*list(l.coords))


def haversine(p_1, p_2):
    '''
        Calculate the haversine distance between two points.

        Input
        =====
            * p_1: The coordinates of the first point (```shapely.geometry.Point``` instance)
            * p_1: The coordinates of the second point (```shapely.geometry.Point``` instance)

        Output
        =====
            * The haversine distance between two points in Kilometers
    '''
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [p_1.x, p_1.y, p_2.x, p_2.y])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.power(np.sin(dlat * 0.5), 2) + np.cos(lat1) * np.cos(lat2) * np.power(np.sin(dlon * 0.5), 2)

    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))


def initial_compass_bearing(point1, point2):
    """
        Calculate the initial compass bearing between two points.

        $ \theta = \atan2(\sin(\Delta lon) * \cos(lat2), \cos(lat1) * \sin(lat2) - \sin(lat1) * \cos(lat2) * \cos(\Delta lon)) $

        Input
        =====
          * point1: shapely.geometry.Point Instance
          * point2: shapely.geometry.Point Instance

        Output
        =====
            The bearing in degrees
    """
    lat1 = np.radians(point1.y)
    lat2 = np.radians(point2.y)
    delta_lon = np.radians(point2.x - point1.x)

    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon))
    initial_bearing = np.arctan2(x, y)

    # Now we have the initial bearing but math.atan2 return values from -180 to + 180
    # Thus, in order to get the range to [0, 360), the solution is:
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


def calculate_velocity(gdf, speed_name='speed', timestamp_name='ts', geometry_name='geom'):
    '''
        Calculate the speed between two points.

        Input
        =====
        +++

        Output
        =====
          The speed in knots (nautical miles per hour)
    '''
    # if there is only one point in the trajectory its velocity will be the one measured from the speedometer
    if len(gdf) == 1:
        return gdf[speed_name]

    speed_attrs = gdf[[speed_name, timestamp_name, geometry_name]].copy()

    # create columns for current and next location. Drop the last columns that contains the nan value
    speed_attrs.loc[:, 'prev_loc']    = speed_attrs[geometry_name].shift()
    speed_attrs.loc[:, 'current_loc'] = speed_attrs[geometry_name]
    speed_attrs.loc[:, 'dt'] 		 = speed_attrs[timestamp_name].diff().abs()

    # get the distance traveled in n-miles and multiply by the rate given (3600/secs for knots) - kilometers to nautical miles
    speed_attrs.loc[:, speed_name] = speed_attrs[['prev_loc', 'current_loc']].iloc[1:]. \
        apply(lambda x : haversine(x[0], x[1]) * 0.539956803456 , axis=1). \
        multiply(3600/speed_attrs.dt)

    # Fill NaN values using the next numeric one
    speed_attrs.loc[:, speed_name].fillna(method='bfill', inplace=True)
    return speed_attrs[speed_name]


def calculate_direction(gdf, course_name='course', geometry_name='geom'):
    '''
        Calculate the Course over Ground (CoG) between two points.

        Input
        =====
        +++

        Output
        =====
          The CoG in degrees ($CoG \in [0, 360)$)
    '''
    # if there is only one point in the trajectory its bearing will be the one measured from the accelerometer
    if len(gdf) == 1:
        return gdf[course_name]

    course_attrs = gdf[[course_name, geometry_name]].copy()

    # create columns for current and next location. Drop the last columns that contains the nan value
    course_attrs.loc[:, 'prev_loc'] 	= course_attrs[geometry_name].shift()
    course_attrs.loc[:, 'current_loc']  = course_attrs[geometry_name]

    course_attrs.loc[:, course_name] = course_attrs[['prev_loc', 'current_loc']].iloc[1:]. \
        apply(lambda x: initial_compass_bearing(x[0], x[1]), axis=1)

    # Fill NaN values using the next numeric one
    course_attrs.loc[:, course_name].fillna(method='bfill', inplace=True)
    return course_attrs[course_name]


def add_speed(gdf, o_id='mmsi', ts='timestamp', speed='speed', geometry='geom', n_jobs=-1, **kwargs):
    # tqdm.tqdm.pandas(**kwargs)
    # gdf.loc[:, speed] = gdf.groupby(o_id, as_index=False, group_keys=False). \
    #     progress_apply(lambda l: calculate_velocity(l.sort_values(ts), speed_name=speed, timestamp_name=ts, geometry_name=geometry))
    gdf.loc[:, speed] = applyParallel(
        gdf.groupby(o_id, as_index=False, group_keys=False), 
        lambda l: calculate_velocity(l.sort_values(ts), speed_name=speed, timestamp_name=ts, geometry_name=geometry), 
        n_jobs=n_jobs,
        **kwargs
    ).reset_index(
        level=list(range(len(o_id))), 
        drop=True
    )
    return gdf


def add_course(gdf, o_id='mmsi', ts='timestamp', course='course', geometry='geom', n_jobs=-1, **kwargs):
    # tqdm.tqdm.pandas(**kwargs)
    # gdf.loc[:, course] = gdf.groupby(o_id, as_index=False, group_keys=False). \
    #     progress_apply(lambda l: calculate_direction(l.sort_values(ts), course_name=course, geometry_name=geometry))
    gdf.loc[:, course] = applyParallel(
        gdf.groupby(o_id, as_index=False, group_keys=False), 
        lambda l: calculate_direction(l.sort_values(ts), course_name=course, geometry_name=geometry), 
        n_jobs=n_jobs,
        **kwargs
    ).reset_index(
        level=list(range(len(o_id))), 
        drop=True
    )
    return gdf


def dms2dec_calc(sign, degree, minute, second):
    return sign * (np.float(degree) + np.float(minute) / 60 + np.float(second) / 3600)


def dms2dec_prep(dms_str):
    """
	# Converting Degrees, Minutes, Seconds formatted coordinate strings to decimal. 

	# Formula:
		> DEC = (DEG + (MIN * 1/60) + (SEC * 1/60 * 1/60))
		> Assumes S/W are negative. 
		
	# Credit: https://gist.github.com/chrisjsimpson/076a82b51e8540a117e8aa5e793d06ec
    

    >>> dms2dec(utf8(48o53'10.18"N))
    48.8866111111F
    
    >>> dms2dec(utf8(2o20'35.09"E))
    2.34330555556F
    
    >>> dms2dec(utf8(48o53'10.18"S))
    -48.8866111111F
    
    >>> dms2dec(utf8(2o20'35.09"W))
    -2.34330555556F
    """
    dms_str = re.sub(r'\s', '', dms_str)
    sign = -1 if re.search('[swSW]', dms_str) else 1

    coords = dms_str.replace('o', 'D ').replace('\'', 'M ').replace('"', 'S').strip('wWsSnNeE ').split(' ')
    dms = {coord[-1]:coord[:-1] for coord in coords}

    degree = dms['D']
    minute = dms['M'] if 'M' in dms else '0'
    second = dms['S'] if 'S' in dms else '0'

    return dms2dec_calc(sign, degree, minute, second)


def getGeoDataFrame_v2(df, coordinate_columns=['lon', 'lat'], crs='epsg:4326'):
    '''
        Create a GeoDataFrame from a DataFrame in a much more generalized form.
    '''

    df.loc[:, 'geom'] = np.nan
    df.geom = df[coordinate_columns].apply(lambda x: Point(*x), axis=1)

    return gpd.GeoDataFrame(df, geometry='geom', crs=crs)


def create_area_grid(spatial_area, crs='epsg:4326', quadrat_width=1000):
    '''
        Segment a spatial area into (an equally spaced) square grid
        
        Input:
            * spatial_area: The area to segment
            * quadrat_width: The squares' width
            
        Output:
            * A GeoSeries containing the grid's squares
            
        Note: the unit of the quadrat_width is in accord to the CRS of the spatial area.
    '''
    # quadrat_width is in the units the geometry is in, so we'll do a tenth of a degree
    geometry_cut = ox.utils_geo._quadrat_cut_geometry(spatial_area, quadrat_width=quadrat_width)

    grid_gdf = gpd.GeoDataFrame(geometry_cut.geoms, columns=['geom'], geometry='geom')
    grid_gdf.crs = crs

    return grid_gdf


def create_area_bounds(spatial_areas, epsg=2154, area_radius=2000):
    '''
    Given some Datapoints, create a circular bound of _area_radius_ kilometers.
    '''
    spatial_areas2 = spatial_areas.copy()
    init_crs = spatial_areas2.crs
    # We convert to a CRS where the distance between two points is returned in meters (e.g. EPSG-2154 (France), EPSG-3310 (North America)),
    # so the buffer function creates a circle with radius _area_radius_ meters from the center point (i.e the port's location point)
    spatial_areas2.geometry = spatial_areas2.geometry.to_crs(epsg=epsg).buffer(area_radius).to_crs(init_crs)
    # After we create the spatial_areas bounding circle we convert back to its previous CRS.
    return spatial_areas2


def classify_area_proximity(trajectories, spatial_areas, o_id_column='id', ts_column='t_msec', area_radius=2000, area_epsg=2154):
    # create the spatial index (r-tree) of the trajectories's data points
    print ('Creating Spatial Index...')
    sindex = trajectories.sindex

    # find the points that intersect with each subpolygon and add them to _points_within_geometry_ DataFrame
    points_within_geometry = []

    if (spatial_areas.geometry.type == 'Point').all():
        spatial_areas = create_area_bounds(spatial_areas, area_radius=area_radius, epsg=area_epsg)

    print ('Classifying Spatial Proximity...')
    for airport_id, poly in spatial_areas.geometry.items():
        possible_matches_index = list(sindex.intersection(poly.bounds))
        possible_matches = trajectories.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(poly)]

        if (len(precise_matches) != 0):
            trajectories.loc[precise_matches.index, 'area_id'] = airport_id
            points_within_geometry.append(trajectories.loc[precise_matches.index])

    print ('Gathering Results...')
    points_within_geometry = pd.concat(points_within_geometry)
    points_within_geometry = points_within_geometry.drop_duplicates(subset=[o_id_column, ts_column])

    # When we create the _traj_id_ column, we label each record with 0,
    # if it's outside the port's radius and -1 if it's inside the port's radius.
    trajectories.loc[trajectories.index.isin(points_within_geometry.index), 'traj_id'] = -1
    trajectories.loc[~trajectories.index.isin(points_within_geometry.index), 'traj_id'] = 0
    trajectories.loc[:, 'label'] = trajectories['traj_id'].values

    return trajectories


def segment_trajectory(df, col_name, threshold=30 * 60, min_pts=10, output_name='traj_nr'):
    delta = df[col_name].diff().abs()

    # Get splitting points
    L = delta.loc[delta >= threshold].index.values.tolist()
    idx = np.cumsum(np.in1d(df.index, L))

    # Drop splitting points
    out = pd.Series(idx, index=df.index, name=output_name, dtype=int)
    out.drop(L, inplace=True)

    # Drop trips with |points| < min_pts (Rejected Trips)
    out2 = out.groupby(out).filter(lambda l: len(l) >= min_pts)
    return out2


def spatial_segmentation(df, col_name, threshold=1, min_pts=10, output_name='traj_nr'):
    traj_nrs = segment_trajectory(df, col_name, threshold, min_pts, output_name)

    df_spat_seg = df.join(traj_nrs, how='inner')
    df_spat_seg.drop(df_spat_seg.index[df_spat_seg[col_name] == -1], inplace=True)

    return df_spat_seg


def temporal_segmentation(df, col_name, threshold=30 * 60, min_pts=10, output_name='traj_nr'):
    traj_nrs = segment_trajectory(df, col_name, threshold, min_pts, output_name)

    df_temp_seg = df.join(traj_nrs, how='inner')
    return df_temp_seg


def temporal_resampling_v2(df, features=['lat', 'lon'], o_id_name='mmsi', temporal_name='ts', temporal_unit='s',
                           rate='60S', method='linear', min_pts=1):
    # The Interpolate module of SciPy requires at least 2 Records
    if len(df) < min_pts:
        return df.iloc[0:0].loc[:, features]

    # Define temp. axis and feature space
    x = pd.to_datetime(df[temporal_name], unit=temporal_unit).values.astype(np.int64)
    y = df[features].values

    # Fetch the starting and ending timestamps of the trajectory
    dt_start = pd.to_datetime(df[temporal_name].min(), unit=temporal_unit)
    dt_end = pd.to_datetime(df[temporal_name].max(), unit=temporal_unit)

    # Interpolate
    f = interp1d(x, y, kind=method, axis=0)
    xnew_V3 = pd.date_range(start=dt_start.round(rate), end=dt_end, freq=rate, inclusive='right')

    # Reconstruct the new (resampled) dataframe
    df_RESAMPLED = pd.DataFrame(f(xnew_V3), columns=features)
    df_RESAMPLED.loc[:, temporal_name] = xnew_V3

    if df_RESAMPLED.empty:
        return df_RESAMPLED

    df_RESAMPLED.loc[:, o_id_name] = getattr(df.iloc[0, :], o_id_name)
    return df_RESAMPLED


def applyParallel(df_grouped, fun, n_jobs=-1, **kwargs):
    '''
    Forked from: https://stackoverflow.com/a/27027632
    '''
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    print(f'Scaling {fun} to {n_jobs} CPUs')

    df_grouped_names = df_grouped.grouper.names

    _fun = lambda name, group: (fun(group.drop(df_grouped_names, axis=1)), name)

    result, keys = zip(*Parallel(n_jobs=n_jobs)(
        delayed(_fun)(name, group) for name, group in tqdm.tqdm(df_grouped, **kwargs)
    ))
    return pd.concat(result, keys=keys, names=df_grouped_names)
