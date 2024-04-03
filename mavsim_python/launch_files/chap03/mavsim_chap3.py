"""
mavsimPy
    - Chapter 2 launch file for Beard & McLain, PUP, 2012
    - Update history:  
        12/27/2018 - RWB
        1/17/2019 - RWB
        1/5/2023 - David L. Christiansen
        7/13/2023 - RWB
"""
import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))
# use QuitListener for Linux or PC <- doesn't work on Mac
#from tools.quit_listener import QuitListener
import numpy as np
import parameters.simulation_parameters as SIM
from message_types.msg_delta import MsgDelta
from models.mav_dynamics import MavDynamics
from viewers.manage_viewers import Viewers

#quitter = QuitListener()
    
# initialize elements of the architecture
mav = MavDynamics(SIM.ts_simulation)
delta = MsgDelta()
# viewers = Viewers(animation=True, data=True)
viewers = Viewers()

# initialize the simulation time
sim_time = SIM.start_time
end_time = 60

# main simulation loop
print("Press 'Esc' to exit...")
while sim_time < end_time:
    # ------- vary forces and moments to check dynamics -------------
    fx = -10  # 10
    fy = 0  # 10
    fz = 0  # 10
    Mx = 0  # 0.1
    My = 0  # 0.1
    Mz = 0  # 0.1
    forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T

    # ------- physical system -------------
    mav.update(forces_moments)  # propagate the MAV dynamics

    # ------- update viewers -------------
    viewers.update(
        sim_time,
        mav.true_state,  # true states
        None,  # estimated states
        None,  # commanded states
        None,  # inputs to aircraft
        None,  # measurements
    )

    # ------- increment time -------------
    sim_time += SIM.ts_simulation

    # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break

# Save an Image of the Plot
viewers.close(dataplot_name="ch3_data_plot", sensorplot_name="")