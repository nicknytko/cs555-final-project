import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import trigauss
import mesh

zx, zw = trigauss.trigauss_boundary(2)
zx8, zw8 = trigauss.trigauss(8)
Nquad = len(zw)

def make_trans(def_tri):
    '''
    Creates the translation matrix from points on the reference triangle
    to points on the deformed (input) triangle.
    
    def_tri - 2x3 array of points
    '''
    return np.array([
        [def_tri[0,1] - def_tri[0,0], def_tri[0,2] - def_tri[0,0]],
        [def_tri[1,1] - def_tri[1,0], def_tri[1,2] - def_tri[1,0]]
    ])

def trans(tri, pts):
    '''
    Translate a set of points from the reference triangle to some
    deformed (input) triangle.
    
    tri - 2x3 array of points defining the triangle vertices.  Should be in CCW order.
    pts - 2xN array of points to deform.
    '''
    T = make_trans(tri)
    return T@pts + (tri[:,0])[:,np.newaxis]

def inv2x2(A):
    '''
    Invert 2x2 matrix.
    '''
    return np.array([[A[1,1], -A[0,1]],[-A[1,0], A[0,0]]])/la.det(A)

def get_triangle(points, tris, i):
    return np.column_stack([
        points[tris[i,j]] for j in range(tris.shape[1])
    ])

def draw_triangles(points, tris):
    '''
    Plot linear triangles.
    '''
    for tri in tris:
        x,y = np.column_stack([
            points[tri[0]], points[tri[1]], points[tri[2]]
        ])
        plt.plot(x,y,'o',markersize=6)
        x = np.append(x,x[0]); y = np.append(y,y[0])
        plt.plot(x,y,'-')

def linear_to_quad_tris(points, triangles):
    '''
    Converts linear triangles to quadratic triangles by inserting points between edges.
    
                      3
    3                 |  \
    |\                6     5     
    | \       ->      |        \
    1--2              1----4----2
    
    points - Nx2 array of points
    triangles - Mx3 array of triangles, indexing into points
    
    returns (qpts, qtris)
    qpts - Nx2 array of points, with new midpoints inserted into end
    qtris - Mx6 array of triangles, indexing into points
    
    Note that triangles will contain same indices between both arrays,
    i.e. triangles[i] and qtri[i] will refer to the same triangle.
    '''
    new_points = points.tolist()
    new_tris = []
    intermed_pts = {}

    def get_intermediate_pt(i,j):
        l = min(i,j); h = max(i,j)
        if l not in intermed_pts:
            intermed_pts[l] = {}
        if h not in intermed_pts[l]:
            ip = (points[l] + points[h])/2
            new_points.append(ip)
            intermed_pts[l][h] = len(new_points)-1
        return intermed_pts[l][h]

    for tri in triangles:
        t1, t2, t3 = tri
        new_tris.append([
            t1, t2, t3,
            get_intermediate_pt(t1, t2),
            get_intermediate_pt(t2, t3),
            get_intermediate_pt(t3, t1)
        ])

    return np.array(new_points), np.array(new_tris, dtype=np.int64)

def draw_triangles_quad(points, tris):
    '''
    Plot quadratic triangles
    '''
    for tri in tris:
        x,y = np.column_stack([
            points[tri[0]], points[tri[3]], points[tri[1]], points[tri[4]], points[tri[2]], points[tri[5]]
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
    triangles - M x P array of triangles, indexing into the points array.
    P depends on the type of triangles (i.e., P=3 for linear, P=6 for quadratic)

    Returns Q, (num global dof x num local dof) sized array
    '''
    N_glob = points.shape[0]
    N_tri = triangles.shape[0]
    tri_len = triangles.shape[1]
    N_local = N_tri * tri_len
    Q = sp.lil_matrix((N_glob, N_local))
    for i in range(N_tri):
        loc_start = tri_len*i
        tri = triangles[i]
        for j in range(tri_len):
            Q[tri[j], loc_start + j] = 1
    return Q.tocsr()

#P1 basis and derivatives
p1_basis = [
    lambda r, s: 1-r-s,
    lambda r, s: r,
    lambda r, s: s,
]
p1_r = [
    lambda r, s: -1,
    lambda r, s: 1,
    lambda r, s: 0,
]
p1_s = [
    lambda r, s: -1,
    lambda r, s: 0,
    lambda r, s: 1,
]

#P2 basis and derivatives
p2_basis = [
    lambda r, s: (1-r-s)*(1-2*r-2*s),
    lambda r, s: r*(2*r-1),
    lambda r, s: s*(2*s-1),
    lambda r, s: 4*r*(1-r-s),
    lambda r, s: 4*r*s,
    lambda r, s: 4*s*(1-r-s),
]
p2_r = [
    lambda r, s: -(1-2*r-2*s)-2*(1-r-s),
    lambda r, s: 4*r-1,
    lambda r, s: 0,
    lambda r, s: 4*(1-r-s) - 4*r,
    lambda r, s: 4*s,
    lambda r, s: 4*s*-1,
]
p2_s = [
    lambda r, s: -(1-2*r-2*s)-2*(1-r-s),
    lambda r, s: 0,
    lambda r, s: 4*s-1,
    lambda r, s: -4*r,
    lambda r, s: 4*r,
    lambda r, s: 4*(1-r-s)-4*s,
]

def create_local_elem_taylor_hood(lpts, ltris, qpts, qtris, e):
    '''
    Create a taylor-hood system, with P2 triangles for velocity and P1 triangles for pressure.
    
    lpts - linear points
    ltris - linear triangles
    qpts - quadratic points
    qtris - quadratic triangles
    e - index into triangles arrays
    
    returns A,B,D1,D2
    '''
    tri_l = get_triangle(lpts, ltris, e)
    tri_q = get_triangle(qpts, qtris, e)
    v_pts = np.array([
        [0.,1.,0.],
        [0.,0.,1.]
    ])
    # compute metrics
    metrics = np.column_stack([compute_deriv_r(tri_l, v_pts), compute_deriv_s(tri_l, v_pts)]) # [dx/dr dx/ds] [dy/dr dy/ds]
    inv_metrics = inv2x2(metrics) # [dr/dx dr/dy] [ds/dx ds/dy]
    drdx = inv_metrics[0,0]; drdy = inv_metrics[0,1]
    dsdx = inv_metrics[1,0]; dsdy = inv_metrics[1,1]
    J = la.det(metrics) #jacobian determinant

    qn = 6
    ln = 3
    Nq = len(zw8)

    A = np.zeros((qn,qn))
    B = np.zeros((qn,qn))

    # A_{ij} = \int d\phi_i/dx d\phi_j/dy dV
    # dp/dx = dp/dr dr/dx + dp/ds ds/dx
    for k in range(Nq):
        r,s = zx8[k]; w = zw8[k]
        for i in range(qn):
            pir = p2_r[i](r,s); pis = p2_s[i](r,s);
            for j in range(qn):
                pjr = p2_r[j](r,s); pjs = p2_s[j](r,s)
                dpix = pir*drdx + pis*dsdx #d\phi_i/dx
                dpjx = pjr*drdx + pjs*dsdx #d\phi_j/dx
                dpiy = pir*drdy + pis*dsdy #d\phi_i/dy
                dpjy = pjr*drdy + pjs*dsdy #d\phi_j/dy
                a = dpix*dpjx*w*J # dv/dx du/dx
                b = dpiy*dpjy*w*J # dv/dy du/dy
                A[i,j] += (a+b)

    # B_{ij} = \int \phi_i \phi_j dV
    for k in range(Nq):
        r,s = zx8[k]; w = zw8[k]
        for i in range(qn):
            pi = p2_basis[i](r,s)
            for j in range(qn):
                pj = p2_basis[j](r,s)
                B[i,j] += pi*pj*w*J

    # (D1)_{ij} = \int \psi_i (d\phi_j/dr dr/dx + d\phi_j/ds ds/dx) dV
    # (D2)_{ij} = \int \psi_i (d\phi_j/dr dr/dy + d\phi_j/ds ds/dy) dV
    D1 = np.zeros((ln,qn))
    D2 = np.zeros((ln,qn))
    for i in range(ln):
        for j in range(qn):
            for k in range(Nq):
                r,s = zx8[k]; w = zw8[k]
                psi_i = p1_basis[i](r,s)
                phi_j_r = p2_r[j](r,s)
                phi_j_s = p2_s[j](r,s)
                D1[i,j] += psi_i * (phi_j_r * drdx + phi_j_s * dsdx) * w * J
                D2[i,j] += psi_i * (phi_j_r * drdy + phi_j_s * dsdy) * w * J

    return A,B,D1,D2

def assemble_taylor_hood_matrices(lpts, ltris):
    '''
    Assemble stiffness and mass matrices given a mesh, using Taylor-Hood elements for
    velocity, pressure.  Used for Stokes flow problems.

    points - N x 2 array of points
    triangles - M x 3 array of triangles, indexing into the points array.

    Returns global A, B, Dx, Dy, qpts, qtris.
    '''
    qpts, qtris = linear_to_quad_tris(lpts, ltris)

    Q_l = create_scatter(lpts, ltris)
    Q_q = create_scatter(qpts, qtris)

    A_L = []
    B_L = []
    D1_L = []
    D2_L = []

    for i in range(ltris.shape[0]):
        A_e, B_e, Dx_e, Dy_e = create_local_elem_taylor_hood(lpts, ltris, qpts, qtris, i)
        A_L.append(A_e)
        B_L.append(B_e)
        D1_L.append(Dx_e)
        D2_L.append(Dy_e)

    A_L = sp.block_diag(A_L)
    B_L = sp.block_diag(B_L)
    D1_L = sp.block_diag(D1_L)
    D2_L = sp.block_diag(D2_L)

    return Q_q@A_L@Q_q.T, Q_q@B_L@Q_q.T, Q_l@D1_L@Q_q.T, Q_l@D2_L@Q_q.T, qpts, qtris


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
    inv_metrics = inv2x2(metrics) # [dr/dx dr/dy] [ds/dx ds/dy]
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
    # divergence operator
    # (D1)_{ij} = \int \psi_i (d\phi_j/dr dr/dx + d\phi_j/ds ds/dx) dV
    # (D2)_{ij} = \int \psi_i (d\phi_j/dr dr/dy + d\phi_j/ds ds/dy) dV
    D1 = np.zeros((3,3))
    D2 = np.zeros((3,3))
    for i in range(3): #row
        for j in range(3): #col
            for k in range(Nquad): #quadrature point
                psi_i = 0
                r,s = zx[k]
                if i==0:
                    psi_i = 1-r-s
                if i==1:
                    psi_i = r
                if i==2:
                    psi_i = s
                phi_j_r = Dr[j]
                phi_j_s = Ds[j]
                D1[i,j] += psi_i * (phi_j_r * inv_metrics[0,0] + phi_j_s * inv_metrics[1,0]) * zw[k] * J
                D2[i,j] += psi_i * (phi_j_r * inv_metrics[0,1] + phi_j_s * inv_metrics[1,1]) * zw[k] * J

    return A, Bh, D1, D2

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
    D1_L = []
    D2_L = []

    for i in range(triangles.shape[0]):
        tri = get_triangle(points, triangles, i)
        A_e, B_e, Dx_e, Dy_e = create_local_elem(tri)
        A_L.append(A_e)
        B_L.append(B_e)
        D1_L.append(Dx_e)
        D2_L.append(Dy_e)

    A_L = sp.block_diag(A_L)
    B_L = sp.block_diag(B_L)
    D1_L = sp.block_diag(D1_L)
    D2_L = sp.block_diag(D2_L)

    return Q@A_L@Q.T, Q@B_L@Q.T, Q@D1_L@Q.T, Q@D2_L@Q.T

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
