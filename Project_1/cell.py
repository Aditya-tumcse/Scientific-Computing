import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
import time
from itertools import product
import utils
from abc import ABCMeta, abstractmethod

class Cell(object, metaclass=ABCMeta):
    
    def __init__(self, lx, ly, r_c, cell_index, neighbor_delta_coordinate, a=1, domain=1):
        """Constructor
        lx : float
            Lower x coordinate
        ly : float
            Lower y coordinate
        r_c : float
            Cut-off radius
        cell_index : int
            Index of cell in list_cells
        neighbor_delta_coordinate : list
            List of  List of numpy array(of size 2), where each numpy array contains the difference between the 
            2d coordinates of current cell and one of neighbor cells in upper right quadrant
        a : int (default value 1)
            Variable lined-cell parameter
        domain : float (default value 1.0)
            Size of domain
        """
        self.side_length = r_c / a
        self.a = a
        self.cell_center = np.array([lx + 0.5 * self.side_length, ly + 0.5 * self.side_length])
        self.cell_index = cell_index
        self.neighbor_cell_index = []
        self.create_neighbor_cell_index(neighbor_delta_coordinate, domain=domain)
                
    def create_neighbor_cell_index(self, neighbor_delta_coordinate, domain=1):
        """Creates neighbor cell index for the current cell
        Parameters
        ----------
        neighbor_delta_coordinate: list
            Relative 2d index of all neighbor interaction cells in first quadrant
        domain: float (Optional value 1.0)
            Size of domain
        """
        self.neighbor_cell_index = []
        ############## Task 1.2 begins ##################
        
        all_neighbor_cell_indices=[]

        # Number of cells in a row in the given domain
        row_length = np.int(np.ceil(domain/self.side_length))

        #Finds lower corner of cell (Recalculated since not taken into account in the constructor)
        lx = self.cell_center[0] - self.side_length * 0.5
        ly = self.cell_center[1] - self.side_length * 0.5

        # Combines the neighbor_delta_coordinate ,horizontal and vertical cells into 
        # overall all_neighbor_cell_indices
        for x in range(1, self.a+1):    
            all_neighbor_cell_indices.append(np.array([x,0]))
            all_neighbor_cell_indices.append(np.array([-x,0]))
            all_neighbor_cell_indices.append(np.array([0,x]))
            all_neighbor_cell_indices.append(np.array([0,-x]))
    
        # Replicate for all 4 quadrants
        for quad_neighbor_ in neighbor_delta_coordinate:
            m,n = quad_neighbor_
            all_neighbor_cell_indices.append(quad_neighbor_)
            all_neighbor_cell_indices.append(np.array([-m,-n]))
            all_neighbor_cell_indices.append(np.array([m,-n]))
            all_neighbor_cell_indices.append(np.array([-m,n]))    

        # Final list makes sure those cells outside the domain are eliminated
        for neighbor_ in all_neighbor_cell_indices:
            rel_x, rel_y = neighbor_
            # Calculate Left Lower coordinates of the cells
            x_cor = lx + rel_x * self.side_length
            y_cor = ly + rel_y * self.side_length
            # Calculate 
            pos_x = np.round(x_cor / self.side_length).astype(np.int)
            pos_y = np.round(y_cor / self.side_length).astype(np.int)
            # Check if boundaries are violated
            if(((pos_x < row_length) and (pos_y < row_length)) and ((pos_x >= 0) and (pos_y >= 0))):
                index = pos_y * row_length + pos_x
                self.neighbor_cell_index.append(index)
                
        ############## Task 1.2 ends ##################
        
    def __str__(self):
        return 'Object of type cell with center {}'.format(self.cell_center)
    
    @abstractmethod
    def add_particle(self, particle_index):
        return
        
    def add_neighbor_cell(self, cell_index):
        self.neighbor_cell_index.append(cell_index)
    
    @abstractmethod
    def delete_all_particles(self):
        return
          
    def plot_cell(self, ax, linewidth=1, edgecolor='r', facecolor='none'):
        lx = self.cell_center[0] - self.side_length * 0.5
        ly = self.cell_center[1] - self.side_length * 0.5
        rect = patches.Rectangle((lx, ly), self.side_length, self.side_length, linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)
   
    @abstractmethod
    def plot_particles(self, list_particles, marker='o', color='r', s=2):
        return
            
    def plot_neighbor_cells(self, ax, list_cells, linewidth=1, edgecolor='r', facecolor='none'):
        for idx in self.neighbor_cell_index:
            list_cells[idx].plot_cell(ax, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)
            
    @abstractmethod
    def plot_neighbor_cell_particles(self, list_cells, list_particles, marker='o', color='r', s=2):
        return
    
    def distance(self, other):
        return np.linalg.norm(self.cell_center - other.cell_center, 2)
    
    def plot_rc(self, ax, rc):
        circle = patches.Circle((self.cell_center[0], self.cell_center[1]), rc)
