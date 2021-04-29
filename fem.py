import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import trigauss
import mesh

zx, zw = trigauss.trigauss(8)
zx2, zw2 = trigauss.trigauss(2)
Nquad = len(zw)

def make_trans(def_tri):
    def_tri = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    return np.array([
        [def_tri[0,1] - def_tri[0,0], def_tri[0,2] - def_tri[0,0]],
        [def_tri[1,1] - def_tri[1,0], def_tri[1,2] - def_tri[1,0]]
    ])

def trans(tri, pts):
    T = make_trans(tri)
    return T@pts + (tri[:,0])[:,np.newaxis]

def inv2x2(A):
    return np.array([[A[1,1], -A[0,1]],[-A[1,0], A[0,0]]])/la.det(A)

def get_triangle(points, tris, i):
    return np.column_stack([
        points[tris[i,0]], points[tris[i,1]], points[tris[i,2]]
    ])

def draw_triangles(points, tris):
    for tri in tris:
        x,y = np.column_stack([
            points[tri[0]], points[tri[1]], points[tri[2]]
        ])
        plt.plot(x,y,'o',markersize=6)
        x = np.append(x,x[0]); y = np.append(y,y[0])
        plt.plot(x,y,'-')

v_pts = np.array([
    [0.,1.,0.],
    [0.,0.,1.]
])

def compute_deriv_r(tri, X):
    Dr = np.array([-1.,1.,0.])
    t_pts = trans(tri, X)
    xr = np.zeros(2)
    for i in range(Dr.shape[0]):
        xr += Dr[i] * t_pts[:,i]
    return xr

def compute_deriv_s(tri, X):
    Ds = np.array([-1.,0.,1.])
    t_pts = trans(tri, X)
    xr = np.zeros(2)
    for i in range(Ds.shape[0]):
        xr += Ds[i] * t_pts[:,i]
    return xr
        
def create_scatter(points, triangles):
    '''
    Creates the operator that maps discontinuous local degrees of freedom to
    their global counterparts.
    
    points - N x 2 array of points
    triangles - M x 3 array of triangles, indexing into the points array.
    
    Returns Q, (num global dof, num local dof) sized array
    '''
    N_glob = points.shape[0]
    N_tri = triangles.shape[0]
    N_local = 3 * N_tri
    Q = sp.lil_matrix((N_glob, N_local))
    for i in range(N_tri):
        loc_start = 3*i
        tri = triangles[i]
        Q[tri[0], loc_start + 0] = 1
        Q[tri[1], loc_start + 1] = 1
        Q[tri[2], loc_start + 2] = 1
    return Q.tocsr()

def create_local_elem(def_tri):
    '''
    Creates the stiffness, mass matrices for a single element.
    
    def_tri - 2 x 3 array of points defining the triangle.
    
    Returns A, B
    '''
    v_pts = np.array([
        [0.,1.,0.],
        [0.,0.,1.]
    ])
    # dX/dR
    metrics = np.column_stack([compute_deriv_r(def_tri, v_pts), compute_deriv_s(def_tri, v_pts)])
    # dR/dX
    inv_metrics = inv2x2(metrics)
    # jacobian determinant
    J = la.det(metrics)
    # integral change of basis
    Rmat = sp.kron(inv_metrics, sp.eye(Nquad))
    # diagonal mass matrix
    Bh = sp.diags([zw * J], [0])
    B = sp.kron(Bh, sp.eye(2))
    # derivative matrices
    Dr = np.array([-1.,1.,0.])
    Drr = np.row_stack([Dr] * Nquad)
    Ds = np.array([-1.,0.,1.])
    Dss = np.row_stack([Ds] * Nquad)
    Dmat = np.vstack([Drr,Dss])
    # full stiffness matrix
    A = Dmat.T @ Rmat.T @ B @ Rmat @ Dmat
    return A, sp.diags([zw2 * J], [0])

def assemble_matrices(points, triangles):
    '''
    Assemble stiffness and mass matrices given a mesh.
    
    points - N x 2 array of points
    triangles - M x 3 array of triangles, indexing into the points array.
    
    Returns global A, B
    '''
    Q = create_scatter(points, triangles)
    A_L = []
    B_L = []
    
    for i in range(triangles.shape[0]):
        tri = get_triangle(points, triangles, i)
        A_e, B_e = create_local_elem(tri)
        A_L.append(A_e)
        B_L.append(B_e)
    
    A_L = sp.block_diag(A_L)
    B_L = sp.block_diag(B_L)
    
    return Q@A_L@Q.T, Q@B_L@Q.T

def create_restriction(N, bcs):
    '''
    Create the restriction operator that enforces Dirichlet conditions by
    removing specific nodes from a matrix.
    
    N - number of global DOF
    bcs - list of nodes to remove
    '''
    R = sp.eye(N).tocsc()
    cols_keep = np.setdiff1d(np.arange(N), bcs, assume_unique=True)
    return R[:,cols_keep]