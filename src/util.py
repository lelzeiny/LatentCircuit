import tkinter as tk
import shapely as sp
from collections import defaultdict

def def_value():
    return []

class Cell:
    def __init__(self, type, id, coords, pins):
        self.shape = sp.Polygon(coords) #shapely object
        self.pins = pins
        self.type = type #type of thing
        self.id = id # number
    

class Wire: #worry about routing later.
    def __init__(self, id, coords):
        self.shape = sp.Polygon(coords)
        self.id = id
    

class Floorplan:
    def __init__(self):
        self.width = 100
        self.height = 100
        self.width_space = 0.5
        self.height_space = 0.5
        self.num_layers = 5

        self.grid = [[[0 for _ in range(self.width)] for _ in range(self.height)] for _ in range(self.num_layers)]
        
        self.cells = {} #empty during inference

        self.wires = [] #empty during inference

    def push_cell(self, ):
        pass

    def push_wire(self, layer, ):
        pass

    def valid_routing(self): #check wires
        pass
    
    def valid_cells(self): #check cells: check area, every instance seen only once
        macros = self.grid[0]
        c = defaultdict(def_value)
        for i in range(self.width):
            for j in range(self.height):
                if isinstance(self.grid[0][i][j], Cell):
                    c[self.grid[0][i][j].id].append((i,j))
        for id in self.cells.keys:
            if c[id]:
                s = sp.polygon(c[id]) #we have  a polygon
                if s.geom_type != 'Polygon' or self.cells[id].shape.length != s.length:
                    return False
            else:
                return False
        return True
            
                    
        

        

if __name__ == "__main__":
    pass #random initialization of a floorplan
