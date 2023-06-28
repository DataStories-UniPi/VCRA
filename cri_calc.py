import numpy as np
import cri_helper as helper
import py.st_toolkit as stt

EPS = 1e-9


def calculate_cpa(own, target):
    """
        Return DCPA, TCPA, etc. (use prior to CRI calculation)

        Outputs
        ==========
        * dist_euclid: The vessels' distance (in NAUTICAL MILES)
        * speed_r: The vessels' relative speed magnitude (in KNOTS)
        * rel_movement_direction: The vessels' relative speed direction (in RADIANS)
        * azimuth_angle_target_to_own: The vessels' azimuth (in RADIANS)
        * relative_bearing_target_to_own: The vessels' relative bearing (in RADIANS)
        * DCPA: The Distance to the Closest Point of Approach ([D]CPA; in NAUTICAL MILES)
        * TCPA: The Time to the Closest Point of Approach ([T]CPA; in HOURS)
    """

    # Calculate D (points' distance)
    dlon, dlat = helper.calculate_delta(own['geometry'].x, target['geometry'].x), \
                 helper.calculate_delta(own['geometry'].y, target['geometry'].y)

    dist_euclid = stt.haversine(own['geometry'], target['geometry']) / 1.852

    # Calculate σ (direction of relative movement direction)
    speed_rx, speed_ry, speed_r = helper.relative_speed(target['speed'], target['course_rad'],
                                                        own['speed'], own['course_rad'])

    rel_movement_direction = helper.azimuth(speed_rx, speed_ry)

    # Calculate $TB_{OS}^{TS}$ (vessels' azimuth)
    azimuth_angle_target_to_own = helper.azimuth(dlon, dlat)

    # Calculate $TB_{OS}^{TS}$ (vessels' azimuth)
    relative_bearing_target_to_own = azimuth_angle_target_to_own - own['course_rad']

    # Calculate CPA
    CPA_angle = rel_movement_direction - azimuth_angle_target_to_own - np.pi
    DCPA = dist_euclid * np.sin(CPA_angle)
    TCPA = dist_euclid * np.cos(CPA_angle) / (speed_r + EPS)

    return dist_euclid, speed_rx, speed_ry, speed_r, \
           rel_movement_direction % (2*np.pi), \
           azimuth_angle_target_to_own  % (2*np.pi), \
           relative_bearing_target_to_own % (2*np.pi), \
           DCPA, TCPA


def calculate_cri(own, target, dist_euclid, speed_r, rel_movement_direction, azimuth_angle_target_to_own,
                  relative_bearing_target_to_own, dcpa, tcpa, stationary_speed_threshold=1, weights=[0.4457, 0.2258, 0.1408, 0.1321, 0.0556]):
    if tcpa <= 0 or own['speed'] <= stationary_speed_threshold:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0

    d1, d2 = helper.calculate_ship_safety_domain(azimuth_angle_target_to_own)
    U_dcpa = helper.cpa_membership(np.abs(dcpa), d1, d2)

    t1, t2 = helper.calculate_collision_eta(np.abs(dcpa), speed_r + EPS, d1, d2)
    U_tcpa = helper.cpa_membership(np.abs(tcpa), t1, t2)

    crit_safe_dist, avoid_measure_dist = helper.calculate_critical_distance(own['length_nmi'], relative_bearing_target_to_own)
    U_dist = helper.cpa_membership(dist_euclid, crit_safe_dist, avoid_measure_dist)

    U_bearing = helper.bearing_membership(relative_bearing_target_to_own)

    speed_ratio = target['speed'] / (own['speed'] + EPS)
    U_speed = helper.speed_ratio_membership(speed_ratio, rel_movement_direction)

    return U_dcpa, U_tcpa, U_dist, U_bearing, U_speed, np.dot(weights, [U_dcpa, U_tcpa, U_dist, U_bearing, U_speed])


def calculate_cri_vcra(own, target, dist_euclid, rel_movement_direction, azimuth_angle_target_to_own, tcpa, 
                       vcra_model, stationary_speed_threshold=1):
    if tcpa <= 0 or own['speed'] <= stationary_speed_threshold:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0
    
    ''' 
        VCRA Input: [
            'own_speed', 'own_course_rad', 
            'target_speed', 'target_course_rad', 
            'dist_euclid', 'azimuth_angle_target_to_own', 'rel_movement_direction'
        ]; Output: CRI [0, 1]
    '''
    vcra_input = np.array(
        [own['speed'], own['course_rad'], target['speed'], target['course_rad'], dist_euclid, azimuth_angle_target_to_own, rel_movement_direction]
    ).reshape(1, -1)

    return np.nan, np.nan, np.nan, np.nan, np.nan, max(0, min(1, vcra_model.predict(vcra_input)[0]))
