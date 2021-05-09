import numpy as np
import pygmsh
import meshio

def load_mesh(file):
    '''
    Load a gmsh file into memory.
    
    file - name of the file to load
    
    returns:
    (points, triangles, boundaries) where:
    points - Nx2 array of point values
    triangles - Mx3 array of point indices
    boundaries - list of arrays, each array consisting of indices along some
      domain boundary
    '''
    mesh = meshio.read(file)
    
    pts = mesh.points
    triangles = []
    boundaries = []
    
    for cell in mesh.cells:
        if cell[0] == 'triangle':
            for tri in cell[1]:
                triangles.append(np.array(tri, dtype=np.int64))
        elif cell[0] == 'line':
            pt_vals = np.unique(cell[1].flatten())
            boundaries.append(pt_vals)
    
    print(f'Loaded mesh with {len(pts)} vertices')
    return pts, np.array(triangles), boundaries