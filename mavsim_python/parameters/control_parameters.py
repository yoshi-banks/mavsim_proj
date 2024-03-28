import numpy as np
import models.model_coef as TF
import parameters.aerosonde_parameters as MAV


#### TODO #####
gravity = MAV.gravity  # gravity constant
Va0 = TF.Va_trim
rho = 0 # density of air
sigma = 0  # low pass filter gain for derivative

#----------roll loop-------------
# get transfer function data for delta_a to phi
wn_roll = 11
zeta_roll = 0.707
a_roll_2 = 130.6
a_roll_1 = 22.6
roll_kp = wn_roll**2 / a_roll_2
roll_kd = (2 * wn_roll * wn_roll - a_roll_1) / a_roll_2

#----------course loop-------------
wn_course = 0.5
zeta_course = 0.707
course_kp = 2 * zeta_course * wn_course * Va0 / gravity
course_ki = wn_course**2 * Va0 / gravity

#----------yaw damper-------------
yaw_damper_p_wo = 0.45
yaw_damper_kr = 0.196

#----------pitch loop-------------
wn_pitch = 15
zeta_pitch = 0.8
a_pitch_1 = 5.288
a_pitch_2 = 99.7
a_pitch_3 = -36.02 
pitch_kp = (wn_pitch**2 - a_pitch_2) / a_pitch_3
pitch_kd = (2 * zeta_pitch * wn_pitch - a_pitch_1) / a_pitch_3
K_theta_DC = pitch_kp * a_pitch_3 / wn_pitch**2

#----------altitude loop-------------
wn_altitude = 0.25
zeta_altitude = 0.8
altitude_kp = (2 * zeta_altitude * wn_altitude) / (K_theta_DC * Va0)
altitude_ki = (wn_altitude**2) / (K_theta_DC * Va0)
altitude_zone = 10

#---------airspeed hold using throttle---------------
wn_airspeed_throttle = 0.5
zeta_airspeed_throttle = 0.707
a_airspeed_throttle_1 = 0.6607
a_airspeed_throttle_2 = 47.02
airspeed_throttle_kp = (2 * zeta_airspeed_throttle * wn_airspeed_throttle - a_airspeed_throttle_1) / a_airspeed_throttle_2
airspeed_throttle_ki = wn_airspeed_throttle**2 / a_airspeed_throttle_2
