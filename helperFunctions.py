import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from glob import glob
from scipy.constants import physical_constants
from scipy.optimize import curve_fit



# gyromagnetic ratio in [rad/s/T]
gamma_p = physical_constants['proton gyromag. ratio'][0]

# vacuum magnetic permeability in [N/A**2]
mu_0 = physical_constants['vacuum mag. permeability'][0]

# field conversion of the dressing field in [uT/Vpp]
fieldConversion = np.array([17.90142671025224, 0.07157374413471178])

# color scheme of lines
lc = ['deepskyblue', 'orange', 'yellowgreen', 'tomato', 'orchid']



def gaussFct(f: np.ndarray, f0: float, A: float, s: float, o: float=0.0) -> np.ndarray:
    '''
    Gaussian function, used to fit the Rabi resonances. 
    '''
    return A * np.exp(-(f-f0)**2/(2*s**2)) + o


def multiGaussFct (f: np.ndarray, *p: float) -> np.ndarray:
    '''
    Multi-Gauss function that is recognized by the number of optimization parameters
    
    Parameters: [f01, A1, s1, f02, A2, s2, ..., offset]
    '''
    n_gauss = (len(p) - 1) // 3
    
    sig = np.zeros_like(f)
    
    for i in range(n_gauss):
        f0, A, s = p[3*i:3*i+3]
        sig += gaussFct(f, f0, A, s)
    return sig + p[-1]


def getF0 (path: str, verbose: int=0) -> tuple[float, float]:
    '''
    Get the Larmor frequency from the spectrum without dressing field
    '''
    # select the undressed spectrum
    data = np.load(path+'dressedStates_00.npz')
    F_SF = data['F_SF']
    Amp = data['Amp']
    
    # fit a Gauss to the spectrum
    popt = (F_SF[Amp[0].argmin()], -2, 30, 1)
    popt, pcov = curve_fit(gaussFct, F_SF, Amp[0], sigma=Amp[1], absolute_sigma=True, p0=popt)
    perr = np.sqrt(np.diag(pcov))

    if verbose>0:
        print(popt)
        print(perr)

    if verbose>1:
        fig, ax = plt.subplots()
        ax.errorbar(F_SF, *Amp, fmt='.')
        ax.plot(F_SF, gaussFct(F_SF, *popt), '-')
        ax.set(xlabel='frequency [Hz]', ylabel='spin polarization')
        plt.show()

    return popt[0], perr[0]


def diagonalizeHamiltonian (x: float, y: float, N: int=50, n: int=0, model: str='QRM', **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Define and diagonalize the Hamiltonian of the Jaynes-Cummings or Quantum-Rabi model
    '''
    if (N-2) % 4 != 0:
        raise ValueError('the following condition has to be fulfilled: (N-2)%4==0')
    
    # create the diagonal elements of H0
    h0_pos = (n + np.arange((N-2)//4, -N//4, -1, dtype=int)) + y/2
    h0_neg = (n + np.arange((N-2)//4, -N//4, -1, dtype=int)) - y/2
    diags = np.vstack((h0_pos,h0_neg)).flatten('F')

    # put the diagonal elements into the H0 matrix
    H0 = np.zeros((N,N))
    for i in range(N):
        H0[i,i] = diags[i]
    
    # create the interaction matrix
    off_1 = np.ones(N-1)/4
    off_1[0::2] = 0
    off_3 = np.ones(N-3)/4
    off_3[1::2] = 0

    # create the interaction matrix depending on the model
    if model=='QRM':
        Vx = (sp.sparse.diags(off_1, offsets=1).toarray()
              + sp.sparse.diags(off_3, offsets=3).toarray() 
              + sp.sparse.diags(off_1, offsets=-1).toarray() 
              + sp.sparse.diags(off_3, offsets=-3).toarray())
    elif model=='JCM':
        Vx = (sp.sparse.diags(off_1, offsets=1).toarray()
              + sp.sparse.diags(off_1, offsets=-1).toarray())
    else:
        raise ValueError('only QRM and JCM are implemented')     
    
    Vx = x * Vx

    # create the full Hamiltonian matrix
    H = H0 + Vx

    # calculate the eigenvalues and eigenvectors of the Hamiltonian
    eigval, eigvec = sp.linalg.eigh(H)

    if 'verbose' in kwargs and kwargs.get('verbose')>0:
        print(f'x\t:\t{x:.3f}')
        print(f'y\t:\t{y:.3f}')
        print(f'N\t:\t{N}')
        
    return eigval, eigvec, H


def calcSpectrum (E: np.ndarray, V: np.ndarray, fd: float, N: int=50, full: bool=False) -> tuple[np.ndarray, np.ndarray]:
    '''
    Calculate the transition frequencies and matrix elements (transition amplitudes) 
    from the eigenvalues and eigenvectors of the diagonalized Hamiltonian
    '''
    # define the probe Hamiltonian
    sigma_x = np.array([[0, 1],[1, 0]])
    V1 = np.kron(np.eye(N//2), sigma_x)

    # Compute all transition amplitudes between dressed states
    if full:
        Fij = fd * (E[None,:] - E[:,None])   # 2D array of freq differences
        Mij = V.conj().T @ (V1 @ V)       # shape (N,N)

        return Fij, Mij

    # Compute only the relevant transition amplitudes between dressed states
    else:
        Fij = np.zeros(N//2)
        Mij = np.zeros(N//2)
        
        for i in np.arange(N//2):
        
            fij = fd * abs(E[N//2+i] - E[N//2-i-1])
            mij = V[:,N//2-i-1].conj().T @ V1 @ V[:,N//2+i]
        
            Fij[i] = fij
            Mij[i] = mij
    
        return Fij, Mij


def calcEnergyLevels_X (X: np.ndarray, y: float, N: int=50, n: int=0, verbose: int=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calculate the energy levels / eigenvalues of the Hamiltonian
    '''
    if (N-2) % 4 != 0:
        raise ValueError('the following condition has to be fulfilled: (N-2)%4==0')

    EV = np.zeros((len(X), N))
    
    for j,x in enumerate(X):

        # create the diagonal elements of H0
        h0_pos = (n + np.arange((N-2)/4, -N/4, -1, dtype=int)) + y/2
        h0_neg = (n + np.arange((N-2)/4, -N/4, -1, dtype=int)) - y/2
        diags = np.vstack((h0_pos,h0_neg)).flatten('F')

        # put the diagonal elements into the H0 matrix
        H0 = np.zeros((N,N))
        for i in range(N):
            H0[i,i] = diags[i]
        
        # create the interaction matrix
        off_1 = np.ones(N-1)/4
        off_1[0::2] = 0
        off_3 = np.ones(N-3)/4
        off_3[1::2] = 0
        Vx = (sp.sparse.diags(off_1, offsets=1).toarray()
              + sp.sparse.diags(off_3, offsets=3).toarray() 
              + sp.sparse.diags(off_1, offsets=-1).toarray() 
              + sp.sparse.diags(off_3, offsets=-3).toarray())
        Vx = x * Vx

        # create the full Hamiltonian matrix
        H = H0 + Vx

        # calculate the eigenvalues of the Hamiltonian
        eigvals = np.linalg.eigvalsh(H)
        EV[j] = eigvals

    # select the n=0 levels to search for crossings
    EV1 = EV[:,N//2]
    EV2 = EV[:,N//2-1]
    
    # find the crossing indices and manually ad the begining and the end
    dE = EV1 - EV2
    idx_peaks = sp.signal.find_peaks(-dE)[0]
    idx_peaks = np.concatenate(([0],idx_peaks+1,[None]))

    # switch energy levels at crossing indices
    EV1_t = np.zeros(len(X))
    EV2_t = np.zeros(len(X))
    for n in np.arange(0, N, 2):
        EV1 = EV[:,n]
        EV2 = EV[:,n+1]
        for i in range(len(idx_peaks)-1):
            if i%2==0:
                EV1_t[idx_peaks[i]:idx_peaks[i+1]] = EV1[idx_peaks[i]:idx_peaks[i+1]]
                EV2_t[idx_peaks[i]:idx_peaks[i+1]] = EV2[idx_peaks[i]:idx_peaks[i+1]]
            else:
                EV1_t[idx_peaks[i]:idx_peaks[i+1]] = EV2[idx_peaks[i]:idx_peaks[i+1]]
                EV2_t[idx_peaks[i]:idx_peaks[i+1]] = EV1[idx_peaks[i]:idx_peaks[i+1]]
        EV[:,n] = EV1_t
        EV[:,n+1] = EV2_t
    
    # flip sign according to |+> to |-> flip
    # this gives a positive resonance frequency for the main resonance at x=0
    sign = np.ones(N//2) if y>1 else -np.ones(N//2)
    sign[0::2] *= -1
    
    if verbose>0:
        print(f'level crossing at \t x = {X[idx_peaks[1]]:.1f}')
        print(f'level anti-crossing at \t x = {X[np.argmin(EV[:,N//2])]:.1f}')
        
    return EV, sign, idx_peaks


def getFitCond (path: str, dataset: int, fitCarrierFrq: bool=False, verbose: int=0) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:

    # get dressing frequency from path
    fd = int(path.split('_')[-1][2:-3])
    
    # load data
    data = np.load(path+'dressedStates_{:02d}.npz'.format(dataset))
    F_SF = data['F_SF']
    Amp = data['Amp']
    vd = data['Vd'][dataset]
    
    # get the baseline
    O0 = Amp[0][F_SF>2700].mean()
    
    # dressing field amplitude in [uT]
    Bd = fieldConversion[0] * vd/1e3
    
    # dressing parameter
    x = (gamma_p/2/np.pi) * Bd / fd / 1e6

    f0 = getF0(path)[0]
    y = f0 / fd
    
    if verbose>0:
        print(f'x: {x:.1f}')
        print(f'Bd: {Bd:.1f} µT')
    
    # calculate the transitions
    E, V, _ = diagonalizeHamiltonian(x, y)
    Fij, Mij = calcSpectrum(E, V, fd)
    Pij = abs(Mij)**2
    
    # get the preliminary resonance amplitude scaling
    S0 = (O0 - Amp[0]).max() / Pij.max()
    
    # resonance width from the measurement in [Hz]
    resWidth = 27
    
    # select significant peaks as initial conditions for the Gaussian fit
    F0 = Fij[Pij>Pij.max()/100]
    A0 = Pij[Pij>Pij.max()/100]

    n_gauss = len(A0)

    if verbose>0:
        print(f'\nnumber of peaks: {len(A0)}\n')
 
    p0 = np.array([[f0, -S0*a0, resWidth] for f0,a0 in zip(F0,A0)]).flatten()
    if fitCarrierFrq:
        p0 = np.concatenate([p0, [fd, -0.05 * (O0 - Amp[0]).max(), resWidth]])
    p0 = np.append(p0, O0)

    # define bounds tuple
    lower = []
    upper = []
    for i in range(n_gauss):
        lower.extend([F0[i]-50, -10, resWidth-10,])
        upper.extend([F0[i]+50, 0, resWidth+10,])

    # bounds for dressing-field nuisance peak
    if fitCarrierFrq:
        lower.extend([fd-1e-6, -10, resWidth-10,])
        upper.extend([fd+1e-6, 10, resWidth+10,])
    
    # offset
    lower.append(-10)
    upper.append(10)
    
    return p0, (np.asarray(lower), np.asarray(upper))


###############################################################################
# PLOT FUNCTIONS
###############################################################################

def plotResonanceFrequencies (path: str, ax: plt.Axes, verbose: int=0, **kwargs: object) -> None:
    '''
    Fit and plot the resonance frequencies in an x-f plot. 
    - Only resonances with an amplitude significance of >= 2 are shown
    - Only resonances where the baseline polarization is >= 10% are shown
    - The dressing-field nuisance peak is fitted if it improves the fit significantly and the amplitude significance is >= 3
    '''

    # get the set dressing field amplitudes in [V]
    dressAmp = np.load(path+'dressedStates_00.npz')['Vd']

    # get dressing frequency from path
    fd = int(path.split('_')[-1][2:-3])
    
    # get actual Larmor frequency from undressed spectrum in [Hz]
    f0 = getF0(path)[0]

    y = f0 / fd

    for i,amp in enumerate(dressAmp):
    
        # load data
        data = np.load(path+'dressedStates_{:02d}.npz'.format(i))
        F_SF = data['F_SF']
        Amp = data['Amp']
        vd = data['Vd'][i]
        
        # dressing field amplitude in [uT]
        Bd = fieldConversion[0]*vd/1e3
    
        # dressing parameter
        x = (gamma_p/2/np.pi) * Bd / fd / 1e6
        
        # calculate the transitions
        E, V, _ = diagonalizeHamiltonian(x, y)
        Fij, Mij = calcSpectrum(E, V, fd)
        Pij = abs(Mij)**2
        P0 = Pij[Pij>Pij.max()/100]

        # Bayesian information criterion
        BIC = np.empty(2)
        for j,carrier in enumerate([False, True]):
        
            p0, bounds = getFitCond(path, i, carrier)
            
            popt, pcov = curve_fit(multiGaussFct, F_SF, Amp[0], sigma=Amp[1], absolute_sigma=True, p0=p0, bounds=bounds)
            perr = np.sqrt(np.diag(pcov))
            residuals = Amp[0] - multiGaussFct(F_SF, *popt)
            chi2 = np.sum((residuals / Amp[1])**2)
            n_data = len(F_SF)
            n_par = len(popt)
            bic = chi2 + n_par * np.log(n_data)
        
            Ac = popt[-3], perr[-3]
            BIC[j] = bic
                    
        dBIC = BIC[1]-BIC[0]
        Asig = abs(Ac[0])/Ac[1]
        if verbose>0:
            print(f'\nDelta BIC: {dBIC:.1f}')
            print(f'Amp. sig.: {Asig:.1f}')
        
        # only if dBIC suggest significant improvement of the model
        # and the dressing-field component is significant
        # include it in the fit
        carrier = dBIC <= -6 and Asig >= 3
        if carrier and verbose>0:
            print('adding dressing-field component\n')
        
        p0, bounds = getFitCond(path, i, carrier)
        popt, pcov = curve_fit(multiGaussFct, F_SF, Amp[0], sigma=Amp[1], absolute_sigma=True, p0=p0, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))

        baselinePolarization = popt[-1]

        # separate Gaussian parameters from the common offset
        fitPars = popt[:-1].reshape(-1, 3)
        fitErrs = perr[:-1].reshape(-1, 3)

        # carrier component is appended last in getFitCond()
        if carrier:
            fitPars = fitPars[:-1]
            fitErrs = fitErrs[:-1]

        resFrq = fitPars[:, 0]
        resAmp = fitPars[:, 1]
        
        resFrqErr = fitErrs[:, 0]
        resAmpErr = fitErrs[:, 1]

        # keep significant resonances
        significance = np.divide(np.abs(resAmp), resAmpErr,  where=resAmpErr>0)
        mask = significance >= 2

        resFrq = resFrq[mask]
        resFrqErr = resFrqErr[mask]

        if baselinePolarization >= 0.1:
            ax.errorbar(np.full_like(resFrq, x), resFrq, resFrqErr, fmt='k.', ms=8)


def plotTransitionFrequencies (ax: plt.Axes, X: np.ndarray, y: float, fd: float, **kwargs: object) -> None:
    '''
    Plot the calculated transition frequencies in an x-f plot. 
    '''
    Ncalc = 50
    Nlevels = 9

    EV, sign, _ = calcEnergyLevels_X(X, y, Ncalc, verbose=1)

    for i in np.arange(Ncalc//2):
        dE = abs(fd *(EV[:,Ncalc//2+i]-EV[:,Ncalc//2-i-1])*sign[i])
        dE -= 2*fd # shift by two periods for proper plotting

        if 'dualColor' in kwargs and kwargs.get('dualColor')==True:
            ax.plot(X, dE, zorder=1, c='{}'.format(lc[4] if i%2==0 else lc[2]), lw=1)
        else:
            ax.plot(X, dE, zorder=1, c='k', lw=2, alpha=0.25)
            ax.plot(X, dE, zorder=1, c='w', lw=1.5, alpha=0.25)
        
        
def plotDensityMap (path: str, ax: plt.Axes, baselineCorr: bool=True, **kwargs: object) -> None:
    '''
    Plot the density map / raw data in an x-f plot. 
    - A baseline correction can be applied to compensate for the loss of the baseline polarization
    '''
    # select the files
    files = sorted(glob(path+'dressedStates_*.npz'))

    # dressing-field amplitudes in [V]
    dressAmp = np.load(path+'dressedStates_00.npz')['Vd']

    # get dressing frequency from path
    fd = int(path.split('_')[-1][2:-3])
    
    # calculate the dressing parameter array
    X = np.linspace(0, dressAmp[-1], 1001) * fieldConversion[0]*gamma_p/2/np.pi/fd/1e9

    # spin-flip field frequency in [Hz]
    F_SF = np.load(files[0])['F_SF']

    # get signal data
    signal = np.zeros((len(files), len(F_SF)))
    for n,file in enumerate(files):
        sig = np.load(file)['Amp'][0]
        
        if baselineCorr:
            sigMean = sig[F_SF>2700].mean()
            signal[n] = sig-sigMean+1
        else:
            signal[n] = sig
            
    signal = np.flip(signal.T, axis=0)

    # plot the density map
    extent=(X.min(), X.max(), 0, F_SF[-1])
    cs = ax.imshow(signal, extent=extent, 
                   aspect='auto', vmin=-0.75, vmax=1.05, cmap='RdBu', interpolation='gaussian')

    # plot the colorbar
    cb = plt.colorbar(cs, pad=0.02, label='spin polarization')