import numpy as np

import shapely
from shapely.ops import transform
from pyproj import Proj, Transformer

EPS = 1e-9


angular_to_nautical_miles = lambda x: shapely.geometry.Point(*[np.divide(i,1852) for i in transform_geometry(x).coords.xy])


def transform_geometry(point, crs_from='epsg:4326', crs_to='epsg:3857'):
	'''
        Transform the CRS of a ```shapely.geometry``` instance
	'''
	project = Transformer.from_crs(crs_from, crs_to, always_xy=True)
	return transform(project.transform, point)


def rectify_direction(xr, yr):
    if xr >= 0 and yr >= 0:
        return 0
    
    elif xr >= 0 and yr < 0:
        return np.pi
    
    elif xr < 0 and yr >= 0:
        return 2 * np.pi
    
    # All negative    
    return np.pi


def calculate_delta(first, second):
    return second - first


def calculate_speed_components(speed, course):
    course_rad = np.deg2rad(course)
    
    return (speed * np.sin(course_rad), 
            speed * np.cos(course_rad))


def calculate_azimuth_angle(xr, yr, eps=EPS):
    # Rectify Azimuth Angle
    beta = rectify_direction(xr, yr)
    
    return np.arctan(xr/(yr + eps)) + beta


def speed_ratio_membership(speed_ratio, relative_course, eps=EPS):
    # Speed Ratio membership 
    denom  = speed_ratio * np.sqrt(speed_ratio**2 + 1 - 2 * speed_ratio * np.sin(np.deg2rad(relative_course)))
    sdenom = 1 / (1 + 2 / (denom + eps))
    
    return sdenom


def relative_course_membership(relative_bearing):
    # Relative Course/Bearing Membership
    x = np.deg2rad(relative_bearing - 19)
    
    return 1/2 * (np.cos(x) + np.sqrt(440/289 + np.cos(x)**2)) - 5/17


def get_safe_domain(vessel_length, relative_bearing):
    x = np.deg2rad(relative_bearing - 19)
    
    d_min = 12 * (vessel_length / 1852)
    d_max = 1.7 * np.cos(x) + np.sqrt(4.4 + 2.89 * np.cos(x)**2)
    
    return d_min, d_max


def normalize(val, v_min, v_max):
    return ((v_max - val) / (v_max - v_min))**2
    

def distance_membership(distance, d_min, d_max):
    # Distance Membership
    if distance < d_min:
        return 1
    elif d_min <= distance <= d_max:
        return normalize(distance, d_min, d_max)
    else:
        return 0


def get_critical_distance(relative_bearing):
    if np.deg2rad(67.5) < relative_bearing <= np.deg2rad(112.5):
        return 1
    elif np.deg2rad(112.5) < relative_bearing <= np.deg2rad(247.5):
        return 0.6
    elif np.deg2rad(247.5) < relative_bearing <= np.deg2rad(355):
        return 0.9
    else:
        return 1.1
    
    
def dcpa_membership(dcpa, d_crit, d_safe):
    # DCPA Membership
    if np.abs(dcpa) < d_crit:
        return 1
    elif d_crit <= np.abs(dcpa) <= d_safe:
        return normalize(np.abs(dcpa), d_crit, d_safe)
    else:
        return 0


def get_response_range(d_crit, d_safe, dcpa, relative_speed, eps=EPS):
    t_crit, t_safe = 0, 0
    
    if np.abs(dcpa) <= d_crit:
        t_crit = np.sqrt(d_crit**2 - dcpa**2) / (relative_speed + eps)
    
    if np.abs(dcpa) <= d_safe:
        t_safe = np.sqrt(d_safe**2 - dcpa**2) / (relative_speed + eps)
    
    return t_crit, t_safe
      
    
def tcpa_membership(tcpa, t_crit, t_safe):    
    # TCPA Membership
    if np.abs(tcpa) < t_crit:
        return 1
    elif t_crit <= np.abs(tcpa) <= t_safe:
        return normalize(np.abs(tcpa), t_crit, t_safe)
    else:
        return 0
