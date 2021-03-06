import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
import time
from itertools import product


def lj_potential(distance, c1=1e-15, c2=1e-5):
    return   (c1 / distance**12) - (c2 / distance**6)

def get_successor_neighbor_delta_coordinate(a=1):
    """Returns neighbor_delta_coordinate
    
    Parameters
    ---------
    a: int
        Variable linked-cell parameter
    """
    neighbor_delta_coordinate = []
    ############# Task 1.1 begins ##################
 
    r = 1 # Consider a cutoff radius of unit size
    # Along a circular radius of 'a' , the distance between top right corner
    # of the cell and bottom left corners are checked if 
    # they are less than the cut-off radius normalized to 1
    for x in range(0, a + 1):
        for y in range(0, a + 1):
            rel_xy_coordinates_ = np.array([x * r /a, y * r /a])
            distance_ = np.linalg.norm(rel_xy_coordinates_, 2)

            if(distance_ < r):
                coordinate_ =  np.array([x + 1, y + 1])
                neighbor_delta_coordinate.append(coordinate_)
  
    ############ Task 1.1 ends #####################
    return neighbor_delta_coordinate

def plot_all_cells(ax, list_cells, edgecolor='r',domain=1):
    for c in list_cells:
        c.plot_cell(ax, edgecolor=edgecolor)
    ax.tick_params(axis='both',labelsize=0, length = 0)
    plt.xlim(left=0, right=domain)
    plt.ylim(bottom=0, top=domain)
    ax.set_aspect('equal', adjustable='box')
    
def get_mean_relative_error(direct_potential, linked_cell_potential):
    return np.mean(np.abs((direct_potential - linked_cell_potential) / direct_potential))
