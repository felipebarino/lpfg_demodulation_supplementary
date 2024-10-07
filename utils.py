import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import logging
from copy import copy
from tqdm import tqdm

logger = logging.getLogger(__name__)


def find_wlres(wl, T, lims=[1.5e-6, 1.6e-6], dwl=3e-9, prominence=0.5):
    """
    Resonant wavelength estimation

    Find the spectrum dip and fit it to a loretzian

    Parameters
    ----------
    wl: np.array
        Wavelength

    T: np.array
        Spectrum
    
    dwl: float
        Wavelength range to fit loretzian

    lims: list
        Resonant wavelength bounds

    prominence: float
        Resonant dip prominence

    Returns
    -------
    wlres: float
        Resonant wavelength
    """
    info = {}

    wl_range = [min(lims)-10e-9, max(lims)+10e-9]
    resolution = np.mean(np.diff(wl))

    mask = ( wl > min(wl_range) ) & (  wl < max(wl_range))
    wl = wl[mask]
    T = T[mask]
    dwl = 3e-9
    resolution_proximity = 3

    peaks, peak_info = find_peaks(-T, prominence=prominence, 
                                  plateau_size=0, wlen=None)
    info = {}
    
    for i in range(len(peaks)):
        wl0 = wl[peaks[i]]
        mask = (wl> wl0 - dwl/2) & (wl < wl0 + dwl/2)

        try:
            popt, _ = curve_fit(transmission_spectra, wl[mask], T[mask],
                                p0=None, max_nfev=10000,
                                bounds=((-np.inf, wl0-resolution_proximity*resolution, 1e-10, -np.inf),
                                        (+np.inf, wl0+resolution_proximity*resolution, 100, np.inf)))

            resonant_wl = popt[1]
            resonant_power = transmission_spectra(popt[1], *popt)

        except RuntimeError:
            resonant_wl = wl[peaks[i]]
            resonant_power = T[peaks[i]]

        if len(peaks) == 1:
            info['resonant_wl'] = resonant_wl
            info['resonant_wl_power'] = resonant_power
        else:
            info[f'resonant_wl_{i}'] = resonant_wl
            info[f'resonant_wl_power_{i}'] = resonant_power

    best_index = np.argmax(peak_info['prominences'])
    info['best_index'] = best_index
    
    try:
        wlres = 1e9*info['resonant_wl']
    except KeyError:
        best = info['best_index']
        wlres = 1e9*info[f'resonant_wl_{best}']
    return wlres


def transmission_spectra(x, a, x0, w, bias):
    """
    Approximates a LPFG spectrum by a loretzian

    Parameters
    ----------
    x: np.array
        Wavelength for simulation

    a: float
        Attenuation intensity

    x0: float
        Resonant wavelength

    w: float
        FWHM

    bias: float
        Insertion loss

    Returns
    -------
    spectrum: np.array
        LPFG array

    """
    return -a*(1 + ((x - x0)/(w/(2*abs(a/3 - 1)**0.5)))**2)**(-1) - bias


def lin_interp(x, y, i, half):
    """
    Linear interpolation
    """
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))


def fwhm(x, y):
    """
    Estimate FWHM

    Parameters
    ----------
    x: np.array
        x-var

    y: np.array
        y-var

    Returns
    -------
    fwhm: float
    """
    half = max(y) - 3.010299956639812
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    crossings = [lin_interp(x, y, zero_crossings_i[0], half),
                 lin_interp(x, y, zero_crossings_i[1], half)]
    return max(crossings) - min(crossings)


def lorentz(x, a, x0, w, b):
    """
    Loretzian function
    """
    return a*(1 + ((x - x0)/(w/2))**2)**(-1) + b


def gaussian(x, a, x0, s, b=0):
    """
    Gaussian function
    """
    arg = -(x - x0)**2 / (2*s**2)
    return a * np.exp(arg) + b


def fbg_reflection(wl_bragg, fwhm, wl, unit='dB'):
    """
    FBG simulation
    Reference on equation:
    @article{peternella2017,
      title={Interrogation of a ring-resonator ultrasound sensor using a fiber Mach-Zehnder interferometer},
      author={Peternella, Fellipe Grillo and Ouyang, Boling and Horsten, Roland and Haverdings, Michael and Kat, Pim and Caro, Jacob},
      journal={Optics express},
      volume={25},
      number={25},
      pages={31622--31639},
      year={2017},
      publisher={Optical Society of America}
    }

    Parameters
    ----------
    wl_bragg: float
        Bragg wavelength
    fwhm: float
        FWHM
    wl: np.array
        Simulation wavelengths
    unit: string
        Unit

    Returns
    -------
    reflection_tf: np.array
        Reflection transfer function
    """
    R = (1 + ((wl - wl_bragg) / (fwhm / 2)) ** 8) ** (-1)
    if unit == 'dB':
        return 10*np.log10(R)
    elif unit == 'linear' or unit == 'lin':
        return R
    else:
        print('Invalid unit')
        return -1


def find_bragg(wl, sp, dwl=6, prominence=9):
    """
    Find Bragg wavelengths

    Parameters
    ----------
    wl: np.array
        Wavelength

    sp: np.array
        Spectrum

    dwl: float
        Wavelength range

    Returns
    -------
    wl_bragg: np.array
        Bragg wavelengths
    peaks: np.array
        Bragg wavelengths' intensities
    """
    diff = np.mean(np.diff(wl))
    loc, info = find_peaks(sp, prominence=prominence, distance=0.48*dwl/diff)
    wl_bragg = []
    peaks = []
    for b_i in loc:
        mask = (wl > wl[b_i]-dwl/2) & (wl < wl[b_i]+dwl/2)
        pars, cov = curve_fit(gaussian, wl[mask], sp[mask], 
                                p0=(-sp[b_i]+min(sp), wl[b_i], 1, -min(sp)), 
                                bounds=((-np.inf, wl[b_i]-dwl, 1e-17, -np.inf), 
                                        (np.inf, wl[b_i]+dwl, 1e+02, np.inf)))
        wl_bragg.append(pars[1])
        peaks.append(gaussian(pars[1], *pars))
    return np.array(wl_bragg), np.array(peaks)


def my_gauss(x, a, x0, w, bias):
    """
    Custom modified gaussian function for LPFG spectrum simulation

    Parameters
    ----------
    x: np.array
        Wavelength for simulation

    a: float
        Attenuation intensity

    x0: float
        Resonant wavelength

    w: float
        FWHM

    bias: float
        Insertion loss

    Returns
    -------
    spectrum: np.array
        LPFG array

    """
    s = 2*(abs(4*np.log(a/3.01)))**0.5
    s = w/s
    arg = -(x - x0)**2 / ((2*s)**2)
    return -a * np.exp(arg) - bias


def arbitrary_funcs(x, a, x0, w, bias, fcn):
    """
    Function to generate synthetic LPFG spectrum using a combination of functions

    Parameters
    ----------
    x: np.array
        Wavelength for simulation

    a: float
        Attenuation intensity

    x0: float
        Resonant wavelength

    w: float
        FWHM

    bias: float
        Insertion loss

    fcn: float
        Function selection parameter

    Returns
    -------
    spectrum: np.array
        LPFG array

    """
    if fcn < 0.2:
        y = transmission_spectra(x, a, x0, w, bias)
    elif fcn < 0.4:
        y = my_gauss(x, a, x0, w, bias)
    elif fcn < 0.6:
        y = 0.5*transmission_spectra(x, a, x0, w, bias) + 0.5*my_gauss(x, a, x0, w, bias)
    elif fcn < 0.8:
        y = -a*transmission_spectra(x, 1, x0, w, 0)*my_gauss(x, 1, x0, w, 0) - bias
    else:
        k = np.random.rand()
        y = k*transmission_spectra(x, a, x0, w, bias) + (1-k)*my_gauss(x, a, x0, w, bias)
    return y


def mapper(x, min_x, max_x, min_y, max_y):
    """
    Map values

    Parameters
    ----------
    x: float
        Input value

    min_x: float
        Minimum value of x

    max_x: float
        Maximum value of x

    min_y: float
        Minimum value of y

    max_y: float
        Maximum value of y

    Returns
    -------
    mapped_value: float
        Mapped value
    """
    dx = max_x-min_x
    dy = max_y-min_y
    return min_y + (dy/dx)*(x-min_x)


def noisy_arbitrary_funcs(x, a, x0, w, bias, fcn):
    """
    Generate a noisy LPFG transmission spectrum

    Parameters
    ----------
    x: np.array
        Wavelength for simulation

    a: float
        Attenuation intensity

    x0: float
        Resonant wavelength

    w: float
        FWHM

    bias: float
        Insertion loss

    fcn: float
        Function selection parameter

    Returns
    -------
    noisy_spectrum: np.array
        Noisy LPFG array

    clean_spectrum: np.array
        Clean LPFG array
    """
    y = arbitrary_funcs(x, copy(a), copy(x0), copy(w), copy(bias), copy(fcn))
    y_clean = copy(y)
    k = mapper(np.random.rand(), 0, 1, 1, 3)
    k = int(np.round(k))
    for i in range(k):
        if k==0:
            n_x0 = mapper(np.random.rand(), 0, 1, 1490e-9, 1610e-09)
            n_a = mapper(np.random.rand(), 0, 1, copy(a)/10, copy(a)/6)
            n_w = mapper(np.random.rand(), 0, 1, 60e-9, 100e-9)
        else:
            n_a = mapper(np.random.rand(), 0, 1, copy(a)/10, copy(a)/4)
            n_w = mapper(np.random.rand(), 0, 1, copy(w)*1.5, 70e-9)
            if np.random.rand() > 0.5:
                if copy(x0)-copy(w)/2 < 1510e-9:
                    n_x0 = mapper(np.random.rand(), 0, 1, copy(x0)+copy(w), 1590e-09)
                else:
                    n_x0 = mapper(np.random.rand(), 0, 1, 1510e-9, copy(x0)-copy(w)/2)
            else:
                if copy(x0)+copy(w)/2 > 1590e-09:
                    n_x0 = mapper(np.random.rand(), 0, 1, 1510e-9, copy(x0)-copy(w))
                else:
                    n_x0 = mapper(np.random.rand(), 0, 1, copy(x0)+copy(w)/2, 1590e-09)
        noisy_peak = arbitrary_funcs(x, n_a, n_x0, n_w, 0, np.random.rand())
        y = y + noisy_peak
        
        if np.random.rand() > 0.6:
            if np.random.rand() > 0.5:
                n_x0 = mapper(np.random.rand(), 0, 1, 1450e-9, 1500e-09)
            else:
                n_x0 = mapper(np.random.rand(), 0, 1, 1600e-9, 1650e-09)
            n_a = mapper(np.random.rand(), 0, 1, copy(a)*0.6, copy(a)*2)
            n_w = mapper(np.random.rand(), 0, 1, copy(w)*0.6, copy(w)*2)
            y = y + arbitrary_funcs(x, n_a, n_x0, n_w, 0, np.random.rand())
    return y, y_clean


def add_bias_to_array(arr):
    """
    Adds a bias to an array by subtracting the maximum value from all elements.

    Parameters:
    arr (numpy.ndarray): The input array.

    Returns:
    numpy.ndarray: The array with the bias added.
    """
    if np.any(arr > 0):
        max_value = np.max(arr)
        bias = -max_value
        arr += bias
    return arr

def correct_bounds(fbg_array, bounds, margin=0.05):
    """
    Corrects the bounds of an FBG array to ensure that they are within the range of the LPFG.

    Parameters:
    - fbg_array (ndarray): The FBG array to correct.
    - bounds (tuple): The lower and upper bounds of the LPFG.
    - margin (int): The percentage margin to add to the bounds.

    Returns:
    - ndarray: The corrected FBG array.
    """
    margin = margin * (bounds[1] - bounds[0])
    fbg_array = np.array(fbg_array)
    mask_lower = fbg_array < bounds[0] + margin
    mask_upper = fbg_array > bounds[1] - margin
    if mask_lower.any():
        fbg_array[mask_lower] = bounds[0] + np.random.uniform(0, margin, sum(mask_lower))
    if mask_upper.any():
        fbg_array[mask_upper] = bounds[1] - np.random.uniform(0, margin, sum(mask_upper))
    return fbg_array

def generate_random_fbg_array(initial_position, variability, n=13):
    """
    Generates a random FBG array within the specified parameters.

    Parameters:
    - initial_position (ndarray): The initial position of each FBG.
    - variability (float): The amount of variability to add to each FBG.
    - n (int): The number of FBGs to generate.

    Returns:
    - ndarray: The generated FBG array.
    """
    array = initial_position + np.random.uniform(-variability, variability, initial_position.shape)
    array = correct_bounds(array, (min(initial_position), max(initial_position)))

    return np.sort(array)

def generate_synth_data(param, N, k, output='dict', quiet=False):
    """
    Generate synthetic data for FBG demodulation.

    Parameters:
    - param (dict): A dictionary of parameter names and their corresponding value ranges.
    - N (int): The number of data samples to generate.
    - k (int): The number of FBG arrays to generate for each data sample.
    - output (str): The output format of the generated data. Options are 'dict' and 'tuple'.
    - quiet (bool): Whether to suppress the progress bar.

    Returns:
    - X (ndarray): An array of noisy LPFG data samples.
    - X_ideal (ndarray): An array of ideal LPFG data samples.
    - wl_bragg (ndarray): An array of Bragg wavelengths for each FBG.
    - wl_res (ndarray): An array of resonance wavelengths for each LPFG.
    """
    if output not in ['dict', 'tuple']:
        raise ValueError("Invalid output format. Please choose either 'dict' or 'tuple'.")
    
    wl_bragg = []
    X = []
    X_ideal = []
    wl_res = []
    base_position = np.linspace(*param['x'], 13)
    dbragg = np.mean(np.diff(base_position))/2

    for i in tqdm(range(N), disable=quiet):
        # generate random parameters
        params = {k: np.random.uniform(*v) for k, v in param.items()}
        for j in range(k):
            # generate random fbg
            if j/k < 0.5:
                variability = dbragg
            else:
                variability = 0.5 * dbragg

            params['x'] = generate_random_fbg_array(base_position, variability, n=13)
            x_noisy, x_ideal = noisy_arbitrary_funcs(**params)
            x_noisy = -add_bias_to_array(x_noisy)
            x_ideal = -add_bias_to_array(x_ideal)

            x_noisy = (x_noisy)/(np.sum(x_noisy))
            x_ideal = (x_ideal)/(np.sum(x_ideal))
            X.append(x_noisy)
            X_ideal.append(x_ideal)
            wl_bragg.append(params['x']*1e9)
            wl_res.append(params['x0']*1e9)
    if output == 'dict':
        return {'input_strength': np.array(X), 
                'input_strength_clean': np.array(X_ideal), 
                'wl_bragg': np.array(wl_bragg), 
                'target': np.array(wl_res)}
    else:
        return np.array(X), np.array(X_ideal), np.array(wl_bragg), np.array(wl_res)

def lorentz_estimation(fbgs, input_strength, full=False, restrict=False, bounds=None, debug=False):
    """
    Estimates the resonant wavelength of an LPFG sensor based on filtered power, using a Lorentzian approximation.

    Parameters:
    fbgs (array-like): An array of wavelengths at which the FBG sensors collect power.
    input_strength (array-like): The corresponding array of input strength.
    full (bool, optional): Determines whether to return all fit parameters or just the resonant wavelength and its uncertainty. Defaults to False.
    restrict (bool, optional): Determines whether to restrict the fit to a certain wavelength range. Defaults to False.
    bounds (tuple, optional): Allows you to specify custom bounds for the fit parameters. If no bounds are provided, default bounds are used.
    debug (bool, optional): Determines whether to plot the fit. Defaults to False.

    Returns:
    tuple: If full is True, returns all fit parameters. If full is False, returns only the resonant wavelength and its uncertainty.
    """
    # Define constants
    # The wavelength range of the proposed interrogator
    WL_RANGE = (1515, 1585)
    # Median distance between FBG Bragg wavelengths
    DELTA_BRAGG = np.median(np.diff(fbgs))
    
    # Initial guess for the peak position is the wavelength with maximum power.
    wl_res_initial = fbgs[np.argmax(input_strength)]
    
    # Adjust initial guess and bounds if restrict is True
    if restrict:
        lower, upper = adjust_bounds(wl_res_initial, DELTA_BRAGG, WL_RANGE)
        bounds, p0 = set_bounds_and_initial_guess(bounds, input_strength, lower, upper, wl_res_initial)
    else:
        wl_res_initial = adjust_initial_guess(wl_res_initial, WL_RANGE)
        bounds, p0 = set_bounds_and_initial_guess(bounds, input_strength, *WL_RANGE, wl_res_initial)
    
    # Fit the data to the Lorentzian function
    try:
        par, pcov = curve_fit(lorentz, fbgs, input_strength, p0=p0, bounds=bounds)
    except Exception as e:
        print(f'Error: {e}\nInput: {(fbgs, input_strength)}\np0={p0}\nbounds={bounds}')
        return None
    
    if debug:
        print(f'Initial guess: {p0}')
        print(f'Bounds: {bounds}')
        print(f'Fit parameters: {par}')
        plt.plot(fbgs, input_strength, 'ok', label='Data')
        wl = np.linspace(min(fbgs), max(fbgs), 1000)
        plt.plot(wl, lorentz(wl, *par), '-r', label='Fit')
        plt.vlines(par[1], 0, par[0], colors='r', linestyles='dashed')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Input strength (arb. units)')

    # Return the fit parameters
    if full:
        return par
    else:
        perr = np.sqrt(np.diag(pcov))
        return par[1], 2*perr[1]


def adjust_bounds(wl_res_initial, delta_bragg, wl_range):
    """
    Adjusts the lower and upper bounds of the wavelength range.

    Parameters:
    wl_res_initial (float): Initial guess for the peak position.
    delta_bragg (float): Median difference between adjacent wavelengths.
    wl_range (tuple): Tuple containing the minimum and maximum wavelength.

    Returns:
    tuple: Adjusted lower and upper bounds of the wavelength range.
    """
    lower = max(wl_res_initial - delta_bragg, min(wl_range))
    upper = min(wl_res_initial + delta_bragg, max(wl_range))
    return lower, upper


def set_bounds_and_initial_guess(bounds, input_strength, lower, upper, wl_res_initial):
    """
    Sets the bounds and initial guess for the curve fitting.

    Parameters:
    bounds (tuple): Tuple containing the lower and upper bounds for the fit parameters.
    input_strength (array-like): The corresponding array of input strength.
    lower (float): Lower bound of the wavelength range.
    upper (float): Upper bound of the wavelength range.
    wl_res_initial (float): Initial guess for the peak position.

    Returns:
    tuple: Tuple containing the bounds and initial guess for the curve fitting.
    """
    if bounds is None:
        bounds = ((max(input_strength), lower,  5, 0), 
                  (1.00, upper, 50, max((min(input_strength)/2, 1e-3))))
        p0 = (1-1e-7, wl_res_initial, 10, 1e-7)
    else:
        p0 = None
    return bounds, p0


def adjust_initial_guess(wl_res_initial, wl_range):
    """
    Adjusts the initial guess for the peak position.

    Parameters:
    wl_res_initial (float): Initial guess for the peak position.
    wl_range (tuple): Tuple containing the minimum and maximum wavelength.

    Returns:
    float: Adjusted initial guess for the peak position.
    """
    if wl_res_initial < min(wl_range): 
        wl_res_initial = min(wl_range)
    if wl_res_initial > max(wl_range):
        wl_res_initial = max(wl_range)
    return wl_res_initial

def get_lpfg_target(wavelength, lpfg_trans):
    """
    This function gets the LPFG resonant wavelength.

    Parameters:
    wavelength (array-like): The wavelength array for the LPFG transmission transfer function, in nm.
    lpfg_trans (array-like): The LPFG transmission transfer function.

    Returns:
    tuple: The normalized difference between the filtered power and its full spectrum, and the normalized FBG positions.
    """
    # The wavelength range of the proposed interrogator
    WL_RANGE = (1515, 1585)
    wl_res = find_wlres(wavelength*1e-9, lpfg_trans)
    if wl_res > max(WL_RANGE) or wl_res < min(WL_RANGE):
        wl_res = np.nan
    return wl_res

def read_anristu_data(filename):
    """
    Read Anristu data from a file and return a pandas DataFrame.

    Parameters:
    filename (str): The path to the file containing the Anristu data.

    Returns:
    pandas.DataFrame: The Anristu data as a DataFrame.

    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if "Wavelength(A)" in line:
                df = pd.read_csv(filename, skiprows=i)
                return df

def interrogate_spectrum(wl, T, lims=[1.5e-6, 1.6e-6], dwl=3e-9, prominence=0.5,return_err=False):
    """
    Resonant wavelength estimation using an LPFG spectrum

    Find the spectrum dip and fit it to a loretzian

    Parameters
    ----------
    wl: np.array
        Wavelength

    T: np.array
        Spectrum
    
    dwl: float
        Wavelength range to fit loretzian

    lims: list
        Resonant wavelength bounds

    prominence: float
        Resonant dip prominence
    
    return_err: bool
        Whether or not to return the resonant wavelength error

    Returns
    -------
    wlres: float
        Resonant wavelength
    """
    info = {}

    wl_range = [min(lims)-10e-9, max(lims)+10e-9]
    resolution = np.mean(np.diff(wl))

    mask = ( wl > min(wl_range) ) & (  wl < max(wl_range))
    wl = wl[mask]
    T = T[mask]
    dwl = 3e-9
    resolution_proximity = 3

    peaks, peak_info = find_peaks(-T, prominence=prominence, 
                                  plateau_size=0, wlen=None)
    info = {}
    
    for i in range(len(peaks)):
        wl0 = wl[peaks[i]]
        mask = (wl> wl0 - dwl/2) & (wl < wl0 + dwl/2)

        try:
            popt, cov = curve_fit(transmission_spectra, wl[mask], T[mask],
                                p0=None, max_nfev=10000,
                                bounds=((-np.inf, wl0-resolution_proximity*resolution, 1e-10, -np.inf),
                                        (+np.inf, wl0+resolution_proximity*resolution, 100, np.inf)))

            resonant_wl = popt[1]
            resonant_wl_err = np.mean(np.diff(wl))/(3)**0.5
            resonant_power = transmission_spectra(popt[1], *popt)

        except RuntimeError:
            resonant_wl = wl[peaks[i]]
            resonant_power = T[peaks[i]]

        if len(peaks) == 1:
            info['resonant_wl'] = resonant_wl
            info['resonant_wl_power'] = resonant_power
            info['resonant_wl_err'] = resonant_wl_err
        else:
            info[f'resonant_wl_{i}'] = resonant_wl
            info[f'resonant_wl_power_{i}'] = resonant_power
            info[f'resonant_wl_err_{i}'] = resonant_wl_err

    best_index = np.argmax(peak_info['prominences'])
    info['best_index'] = best_index
    
    try:
        wlres = 1e9*info['resonant_wl']
        wlres_err = 1e9*info['resonant_wl_err']
    except KeyError:
        best = info['best_index']
        wlres = 1e9*info[f'resonant_wl_{best}']
        wlres_err = 1e9*info[f'resonant_wl_err_{best}']
    if return_err:
        return wlres, wlres_err
    else:
        return wlres