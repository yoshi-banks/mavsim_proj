import numpy as np

def cross(vec: np.ndarray)->np.ndarray:
    return np.array([[0, -vec.item(2), vec.item(1)],
                     [vec.item(2), 0, -vec.item(0)],
                     [-vec.item(1), vec.item(0), 0]])


def S(Theta:np.ndarray)->np.ndarray:
    return np.array([[1,
                      np.sin(Theta.item(0)) * np.tan(Theta.item(1)),
                      np.cos(Theta.item(0)) * np.tan(Theta.item(1))],
                     [0,
                      np.cos(Theta.item(0)),
                      -np.sin(Theta.item(0))],
                     [0,
                      (np.sin(Theta.item(0)) / np.cos(Theta.item(1))),
                      (np.cos(Theta.item(0)) / np.cos(Theta.item(1)))]
                     ])
