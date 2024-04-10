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
from tools.signals import Signals
from models.mav_dynamics_sensors import MavDynamics
from models.wind_simulation import WindSimulation
from controllers.autopilot import Autopilot
#from controllers.lqr_with_rate_damping import Autopilot
# from estimators.observer import Observer
from estimators.observer_ekf import ObserverEKF as Observer
# from estimators.observer_ukf import ObserverUKF as Observer
from viewers.manage_viewers import Viewers

#quitter = QuitListener()

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
autopilot = Autopilot(SIM.ts_simulation)
# observer = Observer(SIM.ts_simulation)
observer = Observer(t0=0)
viewers = Viewers()

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
end_time = 100

# main simulation loop
print("Press 'Esc' to exit...")
while sim_time < end_time:

    # -------autopilot commands-------------
    commands.airspeed_command = Va_command.polynomial(sim_time)
    commands.course_command = chi_command.polynomial(sim_time)
    commands.altitude_command = h_command.polynomial(sim_time)

    # -------- autopilot -------------
    measurements = mav.sensors()  # get sensor measurements

    # estimated_state = observer.update(measurements)  # estimate states from measurements
    estimated_state = observer.update(measurements, sim_time)  # estimate states from measurements

    if sim_time < 1 * end_time:
        delta, commanded_state = autopilot.update(commands, mav.true_state)
    else:
        delta, commanded_state = autopilot.update(commands, estimated_state)

    # -------- physical system -------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------- update viewer -------------
    viewers.update(
        sim_time,
        mav.true_state,  # true states
        estimated_state,  # estimated states
        commanded_state,  # commanded states
        delta,  # inputs to aircraft
        measurements,  # measurements
    )
        
    # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break

    # -------increment time-------------
    sim_time += SIM.ts_simulation

    # time.sleep(SIM.ts_simulation)

# close viewers
viewers.close(dataplot_name="ch8_data_plot", 
              sensorplot_name="ch8_sensor_plot")







