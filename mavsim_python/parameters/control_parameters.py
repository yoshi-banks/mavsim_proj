# import numpy as np
# import models.model_coef as TF
# import parameters.aerosonde_parameters as MAV


# #### TODO #####
# gravity = MAV.gravity  # gravity constant
# Va0 = TF.Va_trim
# rho = 0 # density of air
# sigma = 0  # low pass filter gain for derivative

# #----------roll loop-------------
# # get transfer function data for delta_a to phi
# wn_roll = 11
# zeta_roll = 0.707
# a_roll_2 = 130.6
# a_roll_1 = 22.6
# roll_kp = wn_roll**2 / a_roll_2
# roll_kd = (2 * wn_roll * wn_roll - a_roll_1) / a_roll_2

# #----------course loop-------------
# wn_course = 0.5
# zeta_course = 0.707
# course_kp = 2 * zeta_course * wn_course * Va0 / gravity
# course_ki = wn_course**2 * Va0 / gravity

# #----------yaw damper-------------
# yaw_damper_p_wo = 0.45
# yaw_damper_kr = 0.196

# #----------pitch loop-------------
# wn_pitch = 15
# zeta_pitch = 0.8
# a_pitch_1 = 5.288
# a_pitch_2 = 99.7
# a_pitch_3 = -36.02 
# pitch_kp = (wn_pitch**2 - a_pitch_2) / a_pitch_3
# pitch_kd = (2 * zeta_pitch * wn_pitch - a_pitch_1) / a_pitch_3
# K_theta_DC = pitch_kp * a_pitch_3 / wn_pitch**2

# #----------altitude loop-------------
# wn_altitude = 0.25
# zeta_altitude = 0.8
# altitude_kp = (2 * zeta_altitude * wn_altitude) / (K_theta_DC * Va0)
# altitude_ki = (wn_altitude**2) / (K_theta_DC * Va0)
# altitude_zone = 10

# #---------airspeed hold using throttle---------------
# wn_airspeed_throttle = 0.5
# zeta_airspeed_throttle = 0.707
# a_airspeed_throttle_1 = 0.6607
# a_airspeed_throttle_2 = 47.02
# airspeed_throttle_kp = (2 * zeta_airspeed_throttle * wn_airspeed_throttle - a_airspeed_throttle_1) / a_airspeed_throttle_2
# airspeed_throttle_ki = wn_airspeed_throttle**2 / a_airspeed_throttle_2


# Old parameters
import numpy as np
import models.model_coef as TF
import parameters.aerosonde_parameters as MAV

gravity = MAV.gravity  # gravity constant
rho = MAV.rho  # density of air
sigma = 0.05  # low pass filter gain for derivative
Va0 = TF.Va_trim

#----------roll loop-------------
# get transfer function data for delta_a to phi
wn_roll = 20 #7
zeta_roll = 0.707
roll_kp = wn_roll**2/TF.a_phi2
roll_kd = (2.0 * zeta_roll * wn_roll - TF.a_phi1) / TF.a_phi2

#----------course loop-------------
wn_course = wn_roll / 20.0
zeta_course = 1.0
course_kp = 2 * zeta_course * wn_course * Va0 / gravity
course_ki = wn_course**2 * Va0 / gravity

#----------yaw damper-------------
yaw_damper_p_wo = 0.45  # (old) 1/0.5
yaw_damper_kr = 0.2  # (old) 0.5

#----------pitch loop-------------
wn_pitch = 24.0
zeta_pitch = 0.707
pitch_kp = (wn_pitch**2 - TF.a_theta2) / TF.a_theta3
pitch_kd = (2.0 * zeta_pitch * wn_pitch - TF.a_theta1) / TF.a_theta3
K_theta_DC = pitch_kp * TF.a_theta3 / (TF.a_theta2 + pitch_kp * TF.a_theta3)

#----------altitude loop-------------
wn_altitude = wn_pitch / 30.0
zeta_altitude = 1.0
altitude_kp = 2.0 * zeta_altitude * wn_altitude / K_theta_DC / Va0
altitude_ki = wn_altitude**2 / K_theta_DC / Va0
altitude_zone = 10.0  # moving saturation limit around current altitude

#---------airspeed hold using throttle---------------
wn_airspeed_throttle = 3.0
zeta_airspeed_throttle = 2  # 0.707
airspeed_throttle_kp = (2.0 * zeta_airspeed_throttle * wn_airspeed_throttle - TF.a_V1) / TF.a_V2
airspeed_throttle_ki = wn_airspeed_throttle**2 / TF.a_V2

#---------TECS---------------
tecs_kpE = 10e-3
tecs_kiE = 1e-4
tecs_kpB = 4e-4
tecs_kiB = 1e-5
h_error_max = 10.0