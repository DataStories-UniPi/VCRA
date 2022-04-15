import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.interpolate import interp1d
from shapely.geometry import Point, LineString, shape

import re
import tqdm
import math
from collections import Counter
from multiprocessing import Pool, cpu_count


EARTH_RADIUS = 6371		# Earth's radius in kilometers


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
	speed_attrs.loc[:, speed_name] = speed_attrs[['prev_loc', 'current_loc']].iloc[1:].\
										apply(lambda x : haversine(x[0], x[1]) * 0.539956803456 , axis=1).\
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

	course_attrs.loc[:, course_name] = course_attrs[['prev_loc', 'current_loc']].iloc[1:].\
											apply(lambda x: initial_compass_bearing(x[0], x[1]), axis=1)
	
	# Fill NaN values using the next numeric one
	course_attrs.loc[:, course_name].fillna(method='bfill', inplace=True)
	return course_attrs[course_name]


def add_speed(gdf, o_id='mmsi', ts='timestamp', speed='speed', geometry='geom', **kwargs):
	tqdm.tqdm.pandas(**kwargs)
	gdf.loc[:, speed] = gdf.groupby(o_id, as_index=False, group_keys=False).\
							progress_apply(lambda l: calculate_velocity(l.sort_values(ts), speed_name=speed, timestamp_name=ts, geometry_name=geometry))  
	return gdf


def add_course(gdf, o_id='mmsi', ts='timestamp', course='course', geometry='geom', **kwargs):
	tqdm.tqdm.pandas(**kwargs)
	gdf.loc[:, course] = gdf.groupby(o_id, as_index=False, group_keys=False).\
							 progress_apply(lambda l: calculate_direction(l.sort_values(ts), course_name=course, geometry_name=geometry))  
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
	