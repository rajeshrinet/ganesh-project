# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False

import cython
from libc.math cimport sqrt, pi
from cython.parallel import prange
import numpy as np
cimport numpy as np
DTYPE   = np.float64
ctypedef np.float_t DTYPE_t
from libc.math cimport pow


cdef inline int get_nthreads(int N):
    if N < 50:
        return 1
    elif N < 200:
        return 2
    elif N < 1000:
        return 4
    else:
        return 8

cdef class colloids:
    cdef int particle, N, nthreads
    cdef double xl, b, eta, epsilon, dt, k_spring, s0, bending_strength, mu, muv, bondLength, gamma0, vs, lje, ljr
    cdef readonly np.ndarray drdt, VV, F, S, p, VC
        
    def __init__(self, particle, xl, b, eta, k_spring, vs, bending_strength, bondLength, lje, ljr):
        self.particle = particle;  self.xl = xl;  self.b = b;  self.eta = eta;  
        self.k_spring = k_spring;  
        self.bending_strength = bending_strength
        self.mu = 1/(6*np.pi*self.eta*self.b)
        self.muv = 1/(8*np.pi*self.eta)
        self.N = particle
        self.vs = vs
        self.bondLength = bondLength
        self.gamma0 = 6*np.pi*self.eta*self.b
        self.lje = lje
        self.ljr = ljr
        self.nthreads = get_nthreads(particle)
        
        self.VC = np.zeros( 3*self.particle, dtype=DTYPE)
        self.VV = np.zeros( 3*self.particle, dtype=DTYPE)
        self.F =  np.zeros( 3*self.particle, dtype=DTYPE)
        self.p =  np.zeros( 3*self.particle, dtype=DTYPE)
        self.S =  np.zeros( 5*self.particle, dtype=DTYPE)

            
    cpdef lennardJones(self, double [:] F, double [:] r, double lje, double ljr):
        """
        The standard Lennard-Jones potential truncated at the minimum (aslo called WCA potential)
        
            We choose \phi(r) = lje/12 (rr^12 - 2*rr^6 ) + lje/12,  as the standard WCA potential.
            ljr: minimum of the LJ potential and rr=ljr/r.

        ...

        Parameters
        ----------
        r: np.array
            An array of positions
            An array of size 3*N,
        F: np.array
            An array of forces
            An array of size 3*N,
        lje: float
            Strength of the LJ 
        ljr: float
            Range of the LJ 

        """
        cdef int N = self.N, i, j, Z=2*N
        cdef double dx, dy, dz, dr2, idr, idr2, rminbyr, r6, fac
        cdef double ljr2 = ljr * ljr
        cdef int nthreads = self.nthreads 

        for i in range(N):
            for j in range(i + 1, N):
                dx = r[i   ] - r[j   ]
                dy = r[i+N] - r[j+N]
                dz = r[i+Z] - r[j+Z]
                dr2 = dx*dx + dy*dy + dz*dz

                if dr2 < ljr2:
                    idr = 1.0 / sqrt(dr2)
                    idr2 = idr * idr
                    rminbyr = ljr * idr
                    r6 = rminbyr**6
                    fac = lje * (r6 * (r6 - 1.0)) * idr2
    
                    F[i    ] += fac * dx
                    F[i+N  ] += fac * dy
                    F[i+Z  ] += fac * dz
    
                    F[j    ] -= fac * dx
                    F[j+N  ] -= fac * dy
                    F[j+Z  ] -= fac * dz
               
        return

   
    cdef multipolymers(self, int Nf, double [:] F, double [:] r, double bondLength, double springModulus, double bendModulus, double twistModulus):
        """
        Force on colloids in many polymers connected by a spring 
        
        ...

        Parameters
        ----------
        r: np.array
            An array of positions
        F: np.array
            An array of forces
            An array of size 3*N,
            An array of size 3*N,
        bondLength: float
            The size of natural spring 
        springModulus: float 
            Stiffness of the trap
        """
        cdef int N = self.particle, Z= 2*N
        cdef int nthreads = self.nthreads 
        cdef:
            int i, ii, ip, jp, kp
            int Nm = N/Nf # int/int problem might happen
            double dx12, dy12, dz12, idr12, dx23, dy23, dz23, idr23,
            double F_bend, F_spring, f1, f3, cos1

        # Spring Force
        for ii in prange(Nf,nogil=True,  num_threads = nthreads, schedule = 'guided'):
            for i in range(Nm-1):
                ip = Nm*ii + i
                jp = Nm*ii + i + 1
                dx12 = r[ip]    - r[jp]
                dy12 = r[ip+N] - r[jp+N]
                dz12 = r[ip+Z] - r[jp+Z] #
                F_spring = springModulus*(bondLength/sqrt(dx12*dx12+dy12*dy12+dz12*dz12) - 1.0) # Scalar part of spring force

                F[ip]     += F_spring*dx12
                F[ip + N] += F_spring*dy12
                F[ip + Z] += F_spring*dz12
                F[jp]     -= F_spring*dx12
                F[jp + N] -= F_spring*dy12
                F[jp + Z] -= F_spring*dz12

        # Bending Force
        for ii in prange(Nf,nogil=True,  num_threads = nthreads, schedule = 'guided'):
            for i in range(Nm-2):
                ip = (Nm*ii+i)
                jp = (Nm*ii+i+1)
                kp = (Nm*ii+i+2)
                dx12 = r[ip]   - r[jp]
                dy12 = r[ip+N] - r[jp+N]
                dz12 = r[ip+Z] - r[jp+Z] #
                idr12 = 1.0/sqrt( dx12*dx12 + dy12*dy12 + dz12*dz12 )

                dx23 = r[jp]   - r[kp]
                dy23 = r[jp+N] - r[kp+N]
                dz23 = r[jp+Z] - r[kp+Z] #
                idr23 = 1.0/sqrt( dx23*dx23 + dy23*dy23 + dz23*dz23 )

                cos1 = (dx12*dx23 + dy12*dy23 + dz12*dz23)
                F_bend = bendModulus*idr12*idr23/bondLength

                f1 = F_bend*( dx23 - dx12*cos1*idr12*idr12)
                f3 = F_bend*(-dx12 + dx23*cos1*idr23*idr23)

                F[ip] += f1
                F[jp] -= f1+f3
                F[kp] += f3

                f1 = F_bend*( dy23 - dy12*cos1*idr12*idr12)
                f3 = F_bend*(-dy12 + dy23*cos1*idr23*idr23)

                F[ip+N] += f1
                F[jp+N] -= f1+f3
                F[kp+N] += f3

                f1 = F_bend*( dz23 - dz12*cos1*idr12*idr12)
                f3 = F_bend*(-dz12 + dz23*cos1*idr23*idr23)

                F[ip+Z] += f1
                F[jp+Z] -= f1+f3
                F[kp+Z] += f3
        return

    

    cpdef mobilityTT(self, double [:] v, double [:] r, double [:] F):
        cdef int N  = self.N, i, j, xx=2*N
        cdef double dx, dy, dz, idr, idr2, vx, vy, vz, vv1, vv2, aa = (2.0*self.b*self.b)/3.0 
        cdef double mu=self.mu, muv=self.muv 
        cdef int nthreads = self.nthreads 
        
        for i in prange(N, nogil=True,  num_threads = nthreads, schedule = 'guided'):
            vx=0; vy=0;   vz=0;
            for j in range(N):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr2 = idr*idr
                    
                    vv1 = (1+aa*idr2)*idr 
                    vv2 = (1-3*aa*idr2)*( F[j]*dx + F[j+N]*dy + F[j+xx]*dz )*idr2*idr
                    vx += vv1*F[j]    + vv2*dx 
                    vy += vv1*F[j+N]  + vv2*dy 
                    vz += vv1*F[j+xx] + vv2*dz 

            v[i]    += mu*F[i]    + muv*vx
            v[i+N]  += mu*F[i+N] + muv*vy
            v[i+xx] += mu*F[i+xx] + muv*vz
        return 


    cpdef propulsionT3t_tip(self, double [:] v, double [:] r, double px, double py, double pz, double vs):
        """
        Compute velocity due to 3t mode on the tip: vs*p*(b^3/r^3)(3rr/r^2 - I )` 
        """

        cdef int N = self.N, i, j, xx=2*N  
        cdef double dx, dy, dz, idr, idr3, Ddotidr, vx, vy, vz, mud
        cdef int nthreads = self.nthreads 
 
        for i in prange(N-1, nogil=True,  num_threads = nthreads, schedule = 'guided'):
            # dx = r[i   ] - r[0]
            # dy = r[i+N ] - r[N]
            # dz = r[i+xx] - r[xx] 
            dx = r[i]   - r[N-1]
            dy = r[i+N]  - r[N-1+N]
            dz = r[i+xx] - r[N-1+xx] 
            idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
            mud = 0.5*idr*idr*idr*vs 
            Ddotidr = 3*(px*dx + py*dy + pz*dz)*idr*idr

            v[i]    += mud*(Ddotidr*dx - px)
            v[i+N]  += mud*(Ddotidr*dy - py)
            v[i+xx] += mud*(Ddotidr*dz - pz)

        ## self-propulsion velocity of the tip
        i=N-1
        v[i]    += vs*px
        v[i+N]  += vs*py
        v[i+xx] += vs*pz
        return  

    
    cpdef mobpropulsionT3t_clamp(self, double [:] v, double [:] r, double [:] F, double px, double py, double pz, double vs):
        cdef int N = self.N, i, j, xx=2*N, xx1=3*N, xx2=4*N
        cdef double dx, dy, dz, dr, idr, idr2,  idr3
        cdef double aa=(self.b*self.b*8.0)/3.0, aidr2, mud, b = self.b, mu=self.mu
        cdef double vx, vy, vz, vx1, vy1, vz1, vv1, vv2, Ddotidr, muv=self.muv
        cdef int nthreads = self.nthreads 
 
        for i in prange(2, nogil=True,  num_threads = nthreads, schedule = 'guided'):
            vx=0; vy=0;   vz=0;
            vx1=0; vy1=0;   vz1=0;
            for j in range(N):
                if i != j:
                    dx = r[i]    - r[j]
                    dy = r[i+N] - r[j+N]
                    dz = r[i+xx] - r[j+xx] 
                    idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
                    idr2 = idr*idr
                    
                    vv1 = (1+aa*idr2)*idr 
                    vv2 = (1-3*aa*idr2)*( F[j]*dx + F[j+N]*dy + F[j+xx]*dz )*idr2*idr
                    vx += vv1*F[j]    + vv2*dx 
                    vy += vv1*F[j+N] + vv2*dy 
                    vz += vv1*F[j+xx] + vv2*dz 
            
            dx = r[i]   - r[N-1]
            dy = r[i+N]  - r[N-1+N]
            dz = r[i+xx] - r[N-1+xx] 
            idr = 1.0/sqrt( dx*dx + dy*dy + dz*dz )
            mud = 0.5*vs*idr*idr*idr
            Ddotidr = 3*(px*dx + py*dy + pz*dz)*idr*idr

            v[i]    += mu*F[i]    + muv*vx + mud*(Ddotidr*dx - px)
            v[i+N]  += mu*F[i+N]  + muv*vy + mud*(Ddotidr*dy - py)
            v[i+xx] += mu*F[i+xx] + muv*vz + mud*(Ddotidr*dz - pz)
        return 


    cpdef computeVel_3tClamp(self, double [:] r):
        
        cdef int i, ii, ip,  particle = self.particle, Nf=1, Nm=particle, N=particle, Z=2*N
        cdef double xl = self.xl, b = self.b, eta = self.eta, vs=self.vs, gamma0=self.gamma0, lje = self.lje, ljr = self.ljr 
        cdef double bondLength=self.bondLength, springModulus=self.k_spring, 
        cdef double px, py, pz, bendModulus=self.bending_strength, twistModulus=0
        cdef double [:] VC = self.VC, F=self.F , VV=self.VV 
        

        for i in range(3*particle):
            VV[i] = 0
            VC[i] = 0
            F[i]  = 0 
            
        ip = N-1
        px = r[ip    ] - r[ip - 1]  
        py = r[ip + N] - r[ip + N-1 ] 
        pz = r[ip + Z] - r[ip + Z-1] 

        modp = np.sqrt(px*px + py*py + pz*pz ) + 1e-16
        px   = px/modp
        py   = py/modp
        pz   = pz/modp
        
        self.multipolymers(Nf, F, r, bondLength, springModulus, bendModulus, 0)
        self.lennardJones(F, r, lje, ljr)
        self.mobpropulsionT3t_clamp(VC, r, F, px, py, pz, vs)
        

        F[0]   += -gamma0*VC[0]
        F[1]   += -gamma0*VC[1]
        F[N]   += -gamma0*VC[N]
        F[N+1] += -gamma0*VC[N+1]
        F[2*N] += -gamma0*VC[2*N]
        
        self.mobilityTT(VV, r, F)
        self.propulsionT3t_tip(VV, r, px, py, pz, vs)

        VV[0]  = 0
        VV[1]  = 0
        VV[N]  = 0
        VV[N+1]  = 0
        VV[2*N]  = 0







    








    




            
















    



