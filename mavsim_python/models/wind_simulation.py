"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
from tools.transfer_function import TransferFunction
import numpy as np


class WindSimulation:
    def __init__(self, Ts, gust_flag = True, steady_state = np.array([[0., 0., 0.]]).T):
        # steady state wind defined in the inertial frame
        self._steady_state = steady_state
        ##### TODO #####

        #   Dryden gust model parameters (pg 56 UAV book)

        # HACK:  Setting Va to a constant value is a hack.  We set a nominal airspeed for the gust model.
        # Could pass current Va into the gust function and recalculate A and B matrices.
        Va = 17

        # parameters from Table 4.1
        suv= 1.06
        sw= 0.7
        luv = 200
        lw = 50

        hu_n = np.atleast_2d(np.array([suv*np.sqrt(2*Va/luv)]))
        hu_d = np.atleast_2d(np.array([1, Va/luv]))

        hv_n = np.atleast_2d(suv*np.sqrt(3*Va/luv)*np.array([1, Va/(np.sqrt(3)*luv)]))
        hv_d = np.atleast_2d(np.array([1, 2*Va/luv, (Va/luv)**2]))

        hw_n = np.atleast_2d(sw*np.sqrt(3*Va/lw)*np.array([1, Va/(np.sqrt(3)*lw)]))
        hw_d = np.atleast_2d(np.array([1, 2*Va/lw, (Va/lw)**2]))

        # Dryden transfer functions (section 4.4 UAV book) - Fill in proper num and den
        self.u_w = TransferFunction(num=np.array(hu_n), den=np.array(hu_d),Ts=Ts)
        self.v_w = TransferFunction(num=np.array(hv_n), den=np.array(hv_d),Ts=Ts)
        self.w_w = TransferFunction(num=np.array(hw_n), den=np.array(hw_d),Ts=Ts)

        # self.u_w = TransferFunction(num=np.array([[0]]), den=np.array([[1,1]]),Ts=Ts)
        # self.v_w = TransferFunction(num=np.array([[0,0]]), den=np.array([[1,1,1]]),Ts=Ts)
        # self.w_w = TransferFunction(num=np.array([[0,0]]), den=np.array([[1,1,1]]),Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        return np.concatenate(( self._steady_state, gust ))

