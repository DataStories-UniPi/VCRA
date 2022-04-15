import numpy as np
import cri_helper as helper

EPS = 1e-9


def calculate_azimuth_angle(xr, yr, eps=EPS):
    # Rectify Azimuth Angle
    beta = helper.rectify_direction(xr, yr)
    
    return np.arctan(xr/(yr + eps)) + beta


def colregs_alarms(own, target, eps=EPS):
    '''
    Return DCPA, TCPA, etc. (use prior to CRI calculation)
    '''
    own_geom_nm, target_geom_nm = map(helper.angular_to_nautical_miles, [own['geom'], target['geom']])
    
    # Get vessels' relative position (w.r.t target vessel) -- NAUTICAL MILES
    xr, yr = helper.calculate_delta(own_geom_nm.x, target_geom_nm.x), helper.calculate_delta(own_geom_nm.y, target_geom_nm.y)

    # Get True Azimuth angle (w.r.t target vessel) -- RADIANS
    azimuth_angle = calculate_azimuth_angle(xr, yr, eps=eps)

    # Get vessels' relative Course over Ground (w.r.t target vessel) -- DEGREES
    hr = helper.calculate_delta(own['course'], target['course'])

    # Get vessels' speed components (w.r.t x- and y-axis) -- KNOTS
    speed_tx, speed_ty = helper.calculate_speed_components(target['speed'], target['course'])
    # speed_ox, speed_oy = helper.calculate_speed_components(own['speed'], own['course']+hr)
    speed_ox, speed_oy = helper.calculate_speed_components(own['speed'], own['course'])

    # Get vessels' relative Speed (w.r.t target vessel) -- KNOTS
    speed_rx, speed_ry = helper.calculate_delta(speed_ox, speed_tx), helper.calculate_delta(speed_oy, speed_ty)
    speed_r = np.sqrt(speed_rx**2 + speed_ry**2)

    # Get vessels' Euclidean Distance -- METERS
    dist_euclid = np.sqrt(xr**2 + yr**2)

    # Get vessels' relative Movement angle (w.r.t target vessel) -- RADIANS
    rel_movement_angle = calculate_azimuth_angle(speed_rx, speed_ry, eps=eps)

    # Get vessels' DCPA, and TCPA -- NAUTICAL MILES and HOURS
    ves_angle = rel_movement_angle - azimuth_angle - np.pi
    ves_dcpa = dist_euclid * np.sin(ves_angle)
    ves_tcpa = dist_euclid * np.cos(ves_angle) / (speed_r + EPS)


    return ves_dcpa, ves_tcpa, hr, rel_movement_angle, dist_euclid, speed_r


def calculate_cri(own, target, ves_dcpa, ves_tcpa, hr, rel_movement_angle, dist_euclid, speed_r, weights=[0.4457, 0.2258, 0.1408, 0.1321, 0.0556], eps=EPS):
    speed_ratio = target['speed'] / (own['speed'] + eps)
    # Get vessels' Speed Membership Score
    U_speed = helper.speed_ratio_membership(speed_ratio, rel_movement_angle)
    # U_speed = helper.speed_ratio_membership(speed_ratio, hr)
    
    # Get vessels' Course Membership Score
    U_course = helper.relative_course_membership(hr)
    # U_course = helper.relative_course_membership(rel_movement_angle)

    # Get vessels' Distance Membership Score
    d_min, d_max = helper.get_safe_domain(own['length'], hr)
    # d_min, d_max = helper.get_safe_domain(own['length'], rel_movement_angle)
    U_dist = helper.distance_membership(dist_euclid, d_min, d_max)
    
    # Get vessels' DCPA Membership Score
    d_crit = helper.get_critical_distance(rel_movement_angle)
    # d_crit = helper.get_critical_distance(hr)
    d_safe = 2 * d_crit
    U_dcpa = helper.dcpa_membership(ves_dcpa, d_crit, d_safe)

    # Get vessels' TCPA Membership Score
    t_crit, t_safe = helper.get_response_range(d_crit, d_safe, ves_dcpa, speed_r, eps=eps)
    U_tcpa = helper.tcpa_membership(ves_tcpa, t_crit, t_safe)
    
    # if ves_tcpa < 0 or ves_dcpa < 0:
    if ves_tcpa < 0:
        return 0
    
    return np.dot(weights, [U_dcpa, U_tcpa, U_dist, U_course, U_speed])


if __name__ == '__main__':
    import shapely

    own = dict(
        geom   = shapely.geometry.Point(2.19, 39.4),
        speed  = 30,
        course = 80,
        length = 143
    )  

    target = dict(
        geom   = shapely.geometry.Point(2.7, 39.4),
        speed  = 10,
        course = 190,
        length = 208
    )

    ves_dcpa, ves_tcpa, hr, rel_movement_angle, dist_euclid, speed_r = colregs_alarms(own=own, target=target)
    ves_cri = calculate_cri(own, target, ves_dcpa, ves_tcpa, hr, rel_movement_angle, dist_euclid, speed_r)
    
    print(ves_cri)
