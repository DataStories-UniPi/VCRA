import numpy as np

import shapely
from shapely.ops import transform
from pyproj import Proj, Transformer

EPS = 1e-9

angular_to_nautical_miles = lambda x: shapely.geometry.Point(*[np.divide(i,1852) for i in transform_geometry(x).coords.xy])

degree_geo2math = lambda x: (450 - x) % 360

degrees_to_radians = lambda x: np.deg2rad(x)


def homogenize_units(vcra_data):
    # Unit Conversions (for the sake of homogeneity)

    ## Convert COURSE from DEGREES to RADIANS
    vcra_data.insert(
        vcra_data.columns.get_loc('own_course') + 1, 
        'own_course_rad', 
        vcra_data.own_course.apply(degrees_to_radians)
    )

    vcra_data.insert(
        vcra_data.columns.get_loc('target_course') + 1, 
        'target_course_rad', 
        vcra_data.target_course.apply(degrees_to_radians)
    )


    ## Convert LENGTH from METERS to NAUTICAL MILES
    vcra_data.insert(
        vcra_data.columns.get_loc('own_length') + 1, 
        'own_length_nmi', 
        vcra_data.own_length / 1852
    )

    vcra_data.insert(
        vcra_data.columns.get_loc('target_length') + 1, 
        'target_length_nmi', 
        vcra_data.target_length / 1852
    )
    
    return vcra_data


def transform_geometry(point, crs_from='epsg:4326', crs_to='epsg:3857'):
    '''
        Transform the CRS of a ```shapely.geometry``` instance
    '''
    project = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    return transform(project.transform, point)


def calculate_delta(feat_own, feat_target):
    return feat_target - feat_own


def azimuth(dx, dy):
    return np.arctan2(dx, dy)


def relative_speed(speed_ts, course_ts, speed_os, course_os):
    speed_rx = speed_ts * np.sin(course_ts) - speed_os * np.sin(course_os)
    speed_ry = speed_ts * np.cos(course_ts) - speed_os * np.cos(course_os)
    return speed_rx, speed_ry, np.linalg.norm([speed_rx, speed_ry])


def calculate_delta(first, second):
    return second - first


def calculate_ship_safety_domain(azimuth_angle):
    """
        $d_1$: Ship safety domain
        $d_2$: Ship non-risk boundary

        Input
        =====
          * azimuth_angle (in RADIANS)

        Output
        =====
          * $d_1$, $d_2$
    """
    # ## Calculate $d_1$, and $d_2$. The latter is twice the value of the former,
    # ## therefore, we only need to calculate $d_1$, given the vessels' azimuth angle $TB_{OS}^{TS}$
    d1 = np.nan

    angles = np.array([0, 5*np.pi/8, np.pi, 11*np.pi/8, 2*np.pi])     # FROM THE "Wang et al., 2021" PAPER
    # angles = np.array([0., 112.5, 180., 247.5, 360.])                 # The above parameters, but in degrees.

    if angles[0] <= azimuth_angle < angles[1]:
        d1 = 1.1 - 0.2 * azimuth_angle / np.pi

    elif angles[1] <= azimuth_angle < angles[2]:
        d1 = 1.0 - 0.4 * azimuth_angle / np.pi

    elif angles[2] <= azimuth_angle < angles[3]:
        d1 = 1.0 - 0.4 * (2 * np.pi - azimuth_angle) / np.pi

    elif angles[3] <= azimuth_angle < angles[4]:
        d1 = 1.1 - 0.4 * (2 * np.pi - azimuth_angle) / np.pi

    return d1, 2 * d1


def cpa_membership(value, range_min, range_max):
    if value <= range_min:
        return 1
    elif range_min < value <= range_max:
        return ((range_max - value) / (range_max - range_min))**2
    else:
        return 0


def calculate_collision_eta(dcpa, speed_r, d1, d2):
    nominator = d1 ** 2 - dcpa ** 2

    if dcpa <= d1:
        t1 = np.sqrt(nominator) / speed_r
    else:
        t1 = (d1 - dcpa) / speed_r

    t2 = np.sqrt(d2 ** 2 - dcpa ** 2) / speed_r
    return t1, t2


def calculate_critical_distance(length_own, relative_bearing_target_to_own):
    crit_safe_dist = 12 * length_own

    avoid_measure_angle = relative_bearing_target_to_own - np.deg2rad(19)
    avoid_measure_dist = 1.7 * np.cos(avoid_measure_angle) + np.sqrt(4.4 + 2.89 * np.cos(avoid_measure_angle) ** 2)

    return crit_safe_dist, avoid_measure_dist


def bearing_membership(relative_bearing_target_to_own):
    avoid_measure_angle = relative_bearing_target_to_own - np.deg2rad(19)
    return 1/2 * (np.cos(avoid_measure_angle) + np.sqrt(440/289 + np.cos(avoid_measure_angle)**2)) - 5/17


def speed_ratio_membership(speed_ratio, relative_course):
    denom = speed_ratio * np.sqrt(speed_ratio**2 + 1 + 2 * speed_ratio * np.sin(relative_course)) + EPS
    assert denom > 0, ValueError(f'Division by zero {speed_ratio=}, {relative_course=}, {denom=}')
    return 1/(1 + 2/denom)
