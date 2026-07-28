import numpy as np
from scipy.constants import physical_constants
from time import sleep
from IPython.display import clear_output


######################
# PHYSICAL CONSTANTS #
######################

# just the normal pi = 3.1415
pi = np.pi

# gyromagnetic ratio of the proton in [rad/s/T]
gamma_p = physical_constants['proton gyromag. ratio'][0]


#####################
# GENERAL FUNCTIONS #
#####################

def rabiFct (omega, omega_0, omega_1, tau):
    return omega_1**2 * np.sin(tau/2*np.sqrt((omega-omega_0)**2+omega_1**2))**2 / ((omega-omega_0)**2+omega_1**2)

def gaussFct(f, f0, A, s, o):
    return abs(A) * np.exp(-(f-f0)**2/(2*s**2)) + o

def track_job(job, update_interval=2):
    jobLength = job._number_left
    while job._number_left > 0:
        print('\rTasks (Chunks) remaining = {0}({1}) / {2}({3})     '.format(
        job._number_left*job._chunksize,job._number_left,jobLength*job._chunksize,jobLength), end='')
        sleep(update_interval)
    clear_output()
    print('Task finished')


#########################
# EXPERIMENT PARAMETERS #
#########################

# length of the spin-flip coil in [m]
l_sf = 0.015

# radius of the spin-flip coil in [m]
r_sf = 0.005

# length of the dressing coil in [m]
l_d = 0.06

# radius of the dressing coil in [m]
r_d = 0.015

# measured proton velocity in [m/s]
v_p = 1.6

# main magnetic field amplitude in [T]
B0 = 23.4e-6

# angular frequency of the spinflipper in [rad/s]
omega_SF = gamma_p * B0

# spin-flip field amplitude in [T]
B1 = 2.8e-6

# dressing-field angular frequency in [rad/s]
omega_d = 2 * pi * 350


#########################
# SIMULATION PARAMETERS #
#########################

# number of steps per period of the oscillating fields
Q = 100

# compensation factor to match measured resonance width
vComp = 1.78

# start and end positions of the simulation in [m]
z_0 = -3 * l_d
z_1 = 3 * l_d

# start and end time of the simulation in [s]
t_0 = z_0 / (v_p / vComp)
t_1 = z_1 / (v_p / vComp)

# total propagation time through the setup in [s]
tau = t_1 - t_0

# dressing-field profile (solenoid-like)
Bd_z = lambda z, Bd, L=l_d, R=r_d: Bd * L / 2 * (((z+0.5*L)/(L*np.sqrt(R**2+(z+0.5*L)**2))) - ((z-0.5*L)/(L*np.sqrt(R**2+(z-0.5*L)**2))))

# spin-flip-field profile (solenoid-like)
B1_z = lambda z, B1=B1, L=l_sf, R=r_sf: B1 * L / 2 * (((z+0.5*L)/(L*np.sqrt(R**2+(z+0.5*L)**2))) - ((z-0.5*L)/(L*np.sqrt(R**2+(z-0.5*L)**2))))


########################
# SIMULATION FUNCTIONS #
########################

# spin up representation vector
spinUp = np.array([[1],[0]])

# spin down representation vector
spinDn = np.array([[0],[1]])

def matDressedFlip(t0, t1, z0, B0, B1, omega, theta=0.0, Bd=0, omega_d=0, phi=0, N=1000):

    # calculate the time step
    dt = (t1 - t0) / N
    
    # create the U1 matrix as the identity matrix
    U = np.identity(2, dtype=complex)
    
    # iterate over the time steps
    for i in np.arange(N):
        
        # create the time evolution matrix of this step
        U_step = np.zeros((2,2), dtype=complex)

        # midpoint of time step
        t_mid = t0 + (i+0.5) * dt

        # midpoint position
        z_mid = z0 + (v_p / vComp) * t_mid
        
        # basic calculation, assuming omega_y = 0
        omega_x = -gamma_p * B1_z(z_mid) * np.cos(omega * t_mid + theta) -gamma_p * Bd_z(z_mid, Bd) * np.cos(omega_d * t_mid + phi)
        omega_y = 0
        omega_z = -gamma_p * B0
        
        Omega = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        
        # calculate the propagation matrix of this step
        U_step[0,0] = np.cos(Omega*dt/2) + 1.j*omega_z/Omega*np.sin(Omega*dt/2)
        U_step[0,1] = -1.j*omega_x/Omega*np.sin(Omega*dt/2)
        U_step[1,0] = -1.j*omega_x/Omega*np.sin(Omega*dt/2)
        U_step[1,1] = np.cos(Omega*dt/2) - 1.j*omega_z/Omega*np.sin(Omega*dt/2)

        U = U_step @ U
    
    return U

def dressedResonanceScan(omega_SF, Bd):
        
    result = []
    nTheta = 4
    nPhi = 16
    
    # calculate the number of time steps required
    omega_max = max(abs(gamma_p * B0),
                    abs(omega_SF),
                    abs(omega_d),
                    abs(gamma_p * B1),
                    abs(gamma_p * Bd)
                   )
    dt = 2 * np.pi / (Q * omega_max)
    N = int(np.ceil((t_1 - t_0) / dt))

    result = np.zeros((nTheta, nPhi))
    
    # randomize the phase of the B1 field
    for i,theta in enumerate(np.linspace(0, pi, nTheta, endpoint=False)):
        for j,phi in enumerate(np.linspace(0, 2*pi, nPhi, endpoint=False)):
            result[i,j] = (abs(spinDn.T 
                                 @ matDressedFlip(t_0, t_1, z_0, B0, B1, omega_SF, theta=theta, Bd=Bd, omega_d=omega_d, phi=phi, N=N)
                                 @ spinUp)**2)

    # return result
    return np.mean(result)