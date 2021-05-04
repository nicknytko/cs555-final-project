import numpy as np
import warnings

'''
Gaussian quadrature on reference triangle.
This code is borrowed from Paul Fischer's lecture notes.

(0,1)
  |  \
  |    \
  |      \
(0,0) --- (1,0)
'''

def trigauss_boundary(n):
    '''
    Not really used.  Generates quadrature points on the boundary for a linear triangle.
    These are just the vertex points with appropriate even weighting.
    '''
    if (n==2):
        xw=np.array([
            [0.0, 0.0, 1/3],
            [0.0, 1.0, 1/3],
            [1.0, 0.0, 1/3]])
        qx = xw[:,:2]
        qw = xw[:,2]/2
        return qx, qw
    else:
        return trigauss(n)

def trigauss(n):
    '''
    Generates Gauss points in reference triangular domain.

    n - quadrature degree
    returns points, weights
    '''
    if (n == 1):
        xw=np.array([
            [0.33333333333333, 0.33333333333333, 1.00000000000000]])
    elif (n == 2):
        xw=np.array([
            [0.16666666666667, 0.16666666666667, 0.33333333333333],
            [0.16666666666667, 0.66666666666667, 0.33333333333333],
            [0.66666666666667, 0.16666666666667, 0.33333333333333]])
    elif (n == 3):
        xw=np.array([
            [0.33333333333333, 0.33333333333333, -0.56250000000000],
            [0.20000000000000, 0.20000000000000, 0.52083333333333],
            [0.20000000000000, 0.60000000000000, 0.52083333333333],
            [0.60000000000000, 0.20000000000000, 0.52083333333333]])
    elif (n == 4):
        xw=np.array([
            [0.44594849091597, 0.44594849091597, 0.22338158967801],
            [0.44594849091597, 0.10810301816807, 0.22338158967801],
            [0.10810301816807, 0.44594849091597, 0.22338158967801],
            [0.09157621350977, 0.09157621350977, 0.10995174365532],
            [0.09157621350977, 0.81684757298046, 0.10995174365532],
            [0.81684757298046, 0.09157621350977, 0.10995174365532]])
    elif (n == 5):
        xw=np.array([
            [0.33333333333333, 0.33333333333333, 0.22500000000000],
            [0.47014206410511, 0.47014206410511, 0.13239415278851],
            [0.47014206410511, 0.05971587178977, 0.13239415278851],
            [0.05971587178977, 0.47014206410511, 0.13239415278851],
            [0.10128650732346, 0.10128650732346, 0.12593918054483],
            [0.10128650732346, 0.79742698535309, 0.12593918054483],
            [0.79742698535309, 0.10128650732346, 0.12593918054483]])
    elif (n == 6):
        xw=np.array([
            [0.24928674517091, 0.24928674517091, 0.11678627572638 ],
            [0.24928674517091, 0.50142650965818, 0.11678627572638 ],
            [0.50142650965818, 0.24928674517091, 0.11678627572638 ],
            [0.06308901449150, 0.06308901449150, 0.05084490637021 ],
            [0.06308901449150, 0.87382197101700, 0.05084490637021 ],
            [0.87382197101700, 0.06308901449150, 0.05084490637021 ],
            [0.31035245103378, 0.63650249912140, 0.08285107561837 ],
            [0.63650249912140, 0.05314504984482, 0.08285107561837 ],
            [0.05314504984482, 0.31035245103378, 0.08285107561837 ],
            [0.63650249912140, 0.31035245103378, 0.08285107561837 ],
            [0.31035245103378, 0.05314504984482, 0.08285107561837 ],
            [0.05314504984482, 0.63650249912140, 0.08285107561837]])
    elif (n == 7):
        xw=np.array([
            [0.33333333333333, 0.33333333333333, -0.14957004446768],
            [0.26034596607904, 0.26034596607904, 0.17561525743321 ],
            [0.26034596607904, 0.47930806784192, 0.17561525743321 ],
            [0.47930806784192, 0.26034596607904, 0.17561525743321 ],
            [0.06513010290222, 0.06513010290222, 0.05334723560884 ],
            [0.06513010290222, 0.86973979419557, 0.05334723560884 ],
            [0.86973979419557, 0.06513010290222, 0.05334723560884 ],
            [0.31286549600487, 0.63844418856981, 0.07711376089026 ],
            [0.63844418856981, 0.04869031542532, 0.07711376089026 ],
            [0.04869031542532, 0.31286549600487, 0.07711376089026 ],
            [0.63844418856981, 0.31286549600487, 0.07711376089026 ],
            [0.31286549600487, 0.04869031542532, 0.07711376089026 ],
            [0.04869031542532, 0.63844418856981, 0.07711376089026]])
    elif (n >= 8):
        if(n>8):
            warnings.warn('trigauss: n too high, taking n=8 instead.')
        xw=np.array([
            [0.33333333333333, 0.33333333333333, 0.14431560767779],
            [0.45929258829272, 0.45929258829272, 0.09509163426728],
            [0.45929258829272, 0.08141482341455, 0.09509163426728],
            [0.08141482341455, 0.45929258829272, 0.09509163426728],
            [0.17056930775176, 0.17056930775176, 0.10321737053472],
            [0.17056930775176, 0.65886138449648, 0.10321737053472],
            [0.65886138449648, 0.17056930775176, 0.10321737053472],
            [0.05054722831703, 0.05054722831703, 0.03245849762320],
            [0.05054722831703, 0.89890554336594, 0.03245849762320],
            [0.89890554336594, 0.05054722831703, 0.03245849762320],
            [0.26311282963464, 0.72849239295540, 0.02723031417443],
            [0.72849239295540, 0.00839477740996, 0.02723031417443],
            [0.00839477740996, 0.26311282963464, 0.02723031417443],
            [0.72849239295540, 0.26311282963464, 0.02723031417443],
            [0.26311282963464, 0.00839477740996, 0.02723031417443],
            [0.00839477740996, 0.72849239295540, 0.02723031417443]])
    qx = xw[:,:2]
    qw = xw[:,2]/2
    return qx, qw
