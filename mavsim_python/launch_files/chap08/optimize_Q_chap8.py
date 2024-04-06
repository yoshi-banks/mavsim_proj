"""
mavsim_python
    - Chapter 8 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/21/2019 - RWB
        2/24/2020 - RWB
        1/5/2023 - David L. Christiansen
        7/13/2023 - RWB
        3/11/2024 - RWB
"""
import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
import time
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))
# use QuitListener for Linux or PC <- doesn't work on Mac
#from tools.quit_listener import QuitListener
import numpy as np
import parameters.simulation_parameters as SIM
import parameters.estimation_parameters as EST
from tools.signals import Signals
from models.mav_dynamics_sensors import MavDynamics
from models.wind_simulation import WindSimulation
from controllers.autopilot import Autopilot
#from controllers.lqr_with_rate_damping import Autopilot
# from estimators.observer import Observer
# from estimators.observer_old_old import Observer
# from estimators.observer_old import Observer
from estimators.observer_full import Observer
# from estimators.my_observer import Observer
from viewers.manage_viewers import Viewers
from scipy.optimize import minimize

Nfeval = 1

def compare_states(state, estimate):
    north_max = SIM.north_max
    east_max = SIM.east_max
    altitude_max = SIM.altitude_max
    phi_max = SIM.phi_max
    theta_max = SIM.theta_max
    psi_max = SIM.psi_max
    Va_max = SIM.Va_max
    alpha_max = SIM.alpha_max
    beta_max = SIM.beta_max
    p_max = SIM.p_max
    q_max = SIM.q_max
    r_max = SIM.r_max
    Vg_max = SIM.Vg_max
    gamma_max = SIM.gamma_max
    chi_max = SIM.chi_max
    wn_max = SIM.wn_max
    we_max = SIM.we_max
    bx_max = SIM.bx_max
    by_max = SIM.by_max
    bz_max = SIM.bz_max
    camera_az_max = SIM.camera_az_max
    camera_el_max = SIM.camera_el_max
    error_north = (state.north - estimate.north) / north_max
    error_east = (state.east - estimate.east) / east_max
    error_altitude = (state.altitude - estimate.altitude) / altitude_max
    error_phi = (state.phi - estimate.phi) / phi_max
    error_theta = (state.theta - estimate.theta) / theta_max
    error_psi = (state.psi - estimate.psi) / psi_max
    error_Va = (state.Va - estimate.Va) / Va_max
    error_alpha = (state.alpha - estimate.alpha) / alpha_max
    error_beta = (state.beta - estimate.beta) / beta_max
    error_p = (state.p - estimate.p) / p_max
    error_q = (state.q - estimate.q) / q_max
    error_r = (state.r - estimate.r) / r_max
    error_Vg = (state.Vg - estimate.Vg) / Vg_max
    error_gamma = (state.gamma - estimate.gamma) / gamma_max
    error_chi = (state.chi - estimate.chi) / chi_max
    error_wn = (state.wn - estimate.wn) / wn_max
    error_we = (state.we - estimate.we) / we_max
    error_bx = (state.bx - estimate.bx) / bx_max
    error_by = (state.by - estimate.by) / by_max
    error_bz = (state.bz - estimate.bz) / bz_max
    error_camera_az = (state.camera_az - estimate.camera_az) / camera_az_max
    error_camera_el = (state.camera_el - estimate.camera_el) / camera_el_max
    # find mse of errors
    obj = error_north**2 + error_east**2 + error_altitude**2 + error_phi**2 + error_theta**2 + error_psi**2 + error_Va**2 + error_alpha**2 + error_beta**2 + error_p**2 + error_q**2 + error_r**2 + error_Vg**2 + error_gamma**2 + error_chi**2 + error_wn**2 + error_we**2 + error_bx**2 + error_by**2 + error_bz**2 + error_camera_az**2 + error_camera_el**2
    return obj

def compare_states_list(true_state_list, estimated_state_list):
    # compare estimated to true states
    obj = 0
    for state, estimate in zip(true_state_list, estimated_state_list):
        obj += compare_states(state, estimate)
    return obj
        
def obj_fun(x):

    #quitter = QuitListener()

    # initialize elements of the architecture
    wind = WindSimulation(SIM.ts_simulation)
    mav = MavDynamics(SIM.ts_simulation)
    autopilot = Autopilot(SIM.ts_simulation)
    observer = Observer(SIM.ts_simulation, x)
    # viewers = Viewers()

    # autopilot commands
    from message_types.msg_autopilot import MsgAutopilot
    commands = MsgAutopilot()
    Va_command = Signals(dc_offset=25.0,
                        amplitude=3.0,
                        start_time=2.0,
                        frequency = 0.01)
    h_command = Signals(dc_offset=100.0,
                        amplitude=20.0,
                        start_time=0.0, 
                        frequency=0.02)
    chi_command = Signals(dc_offset=np.radians(0.0),
                        amplitude=np.radians(45.0),
                        start_time=10.0,
                        frequency=0.015)

    # initialize the simulation time
    sim_time = SIM.start_time
    end_time = 10

    true_state_list = []
    estimated_state_list = []
    # main simulation loop
    print("Press 'Esc' to exit...")
    while sim_time < end_time:

        # -------autopilot commands-------------
        commands.airspeed_command = Va_command.polynomial(sim_time)
        commands.course_command = chi_command.polynomial(sim_time)
        commands.altitude_command = h_command.polynomial(sim_time)

        # -------- autopilot -------------
        measurements = mav.sensors()  # get sensor measurements
        # estimated_state = observer.update(measurements, sim_time, mav.true_state)  # estimate states from measurements
        estimated_state = observer.update(measurements)  # estimate states from measurements
        # delta, commanded_state = autopilot.update(commands, estimated_state)
        delta, commanded_state = autopilot.update(commands, mav.true_state)

        # -------- physical system -------------
        current_wind = wind.update()  # get the new wind vector
        mav.update(delta, current_wind)  # propagate the MAV dynamics

        # -------- update viewer -------------
        # viewers.update(
        #     sim_time,
        #     mav.true_state,  # true states
        #     estimated_state,  # estimated states
        #     commanded_state,  # commanded states
        #     delta,  # inputs to aircraft
        #     measurements,  # measurements
        # )
        true_state_list.append(mav.true_state)
        estimated_state_list.append(estimated_state)
            
        # -------Check to Quit the Loop-------
        # if quitter.check_quit():
        #     break

        # -------increment time-------------
        sim_time += SIM.ts_simulation

        # time.sleep(SIM.ts_simulation)
    obj = compare_states_list(true_state_list, estimated_state_list)
    return obj

def callback_fn(Xi):
    global Nfeval
    print('{0:4d}'.format(Nfeval))
    print(Xi.reshape(14,14))
    Nfeval += 1

initial_guess = EST.Q.flatten()
bounds = [(0, 10) for i in range(14*14)]
print(bounds)

# method='L-BFGS-B'
result = minimize(obj_fun, initial_guess, bounds=bounds, callback=callback_fn, method='SLSQP')

print("optimal solution:", result.x)
print("optimal value:", result.fun)

# close viewers
# viewers.close(dataplot_name="ch8_data_plot", 
#               sensorplot_name="ch8_sensor_plot")







