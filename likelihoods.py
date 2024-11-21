"""
This program contains a set of likelihood functions of whitened reconstructed signal
either in time-domain or frequency domain. Each likelihood function is associated with
the basic bilby likelihood class to call a sampler.
"""
from __future__ import division
import numpy as np
import scipy as sp
import lal
import lalsimulation as lalsim
import bilby

__author__ = "Soumen Roy <soumen.roy@ligo.org>"


class StandardLikelihoodTDForSingleImage(bilby.Likelihood):
    def __init__(self, wf_dict, wf_dictL, trigTimeU, trigTimeL,  deltaT):
        """
        A standard likelihood for single lensed image.

        Parameters
        ----------
        wf_dict: dict
            Dictonary of the unlensed event which in each members key is
            the detector name and corresponding value is a subdictonary.
            The members of each subdictonary contains the time samples and
            reconstructed whiten strain data.
            
        wf_dictL: dict
            Dictonary of the lensed image event which in each members key is
            the detector name and corresponding value is a subdictonary.
            The members of each subdictonary contains the time samples and
            reconstructed whiten strain data.
            
        trigTimeU: float
            Trigger time of the event. 
        trigTimeL: float
            Trigger time of the ;lensed event.
         deltaT: float
             Inverse sampling rate of the time domain data.
             
        Init parameters
        ---------------
        mu: float
            The value of magnification factor.
        time: float
            GPS time of unlensed event in the geocentric frame.
        timeL: float
            GPS time of lensed event in the geocentric frame.
        psi: float
            Polarization angle of the unlensed event.
        psiL: float
            Polarization angle of the lensed event.
        ra: float 
            Right ascension of the source in radians
        dec: float
            Declination of the source in radians
        """
        super().__init__(parameters={'mu': None, 'time': None, 'timeL': None, 'psi': None, 'psiL': None, \
                             'ra': None, 'dec': None})
        
        self._wf_dict = wf_dict
        self._wf_dictL = wf_dictL
        self._det_list = list( wf_dict.keys() )
        self._det_listL = list( wf_dictL.keys() )
    
        self._Nd = len( self._det_list )
        self._NdL = len( self._det_listL )
        self._totalNd = self._Nd + self._NdL
        self.deltaT = deltaT
        self.sampling_rate = int(1.0/deltaT)
    
        self._timeU = trigTimeU
        self._timeL = trigTimeL
        
        self._noise_log_likelihood =  np.sum( [np.sum(np.abs(item['strain'])**2) for det, item in wf_dict.items()] ) + \
                    np.sum( [ np.sum(np.abs(item['strain'])**2) for det, item in wf_dictL.items()] )
        
    
    def _calculate_gw_projection_matrix(self):
        """
        This function calculates a projection operator that projects the data
        into the subspace spanned by plus polarization (F+) and cros polarization (Fx).
        """
        
        mu = self.parameters['mu']
        time = self.parameters['time']
        timeL = self.parameters['timeL']
        ra = self.parameters['ra']
        dec = self.parameters['dec']
        psi = self.parameters['psi']
        psiL = self.parameters['psiL']
        
        mu = mu**0.5
        
        Amat = np.zeros(( self._totalNd, 2  ))
        for ii in range( self._Nd ):
            det = self._det_list[ii]
            detector = lal.cached_detector_by_prefix[ det ]
            Fp, Fc = lal.ComputeDetAMResponse(detector.response, ra, \
                                        dec, psi, lal.GreenwichMeanSiderealTime(time))
            Amat[ii] = [Fp, Fc]
        
        for jj in range( self._NdL ):
            det = self._det_listL[jj]
            detector = lal.cached_detector_by_prefix[ det ]
            Fp, Fc = lal.ComputeDetAMResponse(detector.response, ra, \
                                        dec, psiL, lal.GreenwichMeanSiderealTime(timeL))
            Amat[self._Nd+jj] = [mu*Fp, mu*Fc]
            
        Amat = np.matrix(Amat)
        P = np.matmul( Amat, np.matmul( np.matmul( Amat.T, Amat ).I, Amat.T))
        return np.array(P)
    
    
    def _construct_strain_arr(self):  
        """
        This function construct an totalNd x Ns array for the time-domain strains,
        where totalNd is the total number of strains and Ns is number of samples in
        a strain. Each strain array of the 2D array is time shifted according to the
        sky location and proposed time.
        """
        time = self.parameters['time']
        timeL = self.parameters['timeL']
        ra = self.parameters['ra']
        dec = self.parameters['dec']
        
        strain_arr = []
        
        for ii in range( self._Nd ):
            det = self._det_list[ii]
            t_delay = lal.TimeDelayFromEarthCenter( \
                            lalsim.DetectorPrefixToLALDetector( det ).location, ra, dec, time)
            strain_arr.append( np.roll( self._wf_dict[det]['strain'], \
                                       int( (time - self._timeU -t_delay)*self.sampling_rate ) ) )
            
        for jj in range( self._NdL ):
            det = self._det_listL[jj]
            t_delay = lal.TimeDelayFromEarthCenter( \
                            lalsim.DetectorPrefixToLALDetector( det ).location, ra, dec, timeL)
            strain_arr.append( np.roll( self._wf_dictL[det]['strain'], \
                                       int( (timeL - self._timeL -t_delay)*self.sampling_rate ) ) )
            
        strain_arr = np.array(strain_arr)
        return strain_arr
    
    
    def noise_log_likelihood(self):
        """
        Calculates the real part of noise log-likelihood
        
        Returns
        -------
        float: The noise log likelihood value
        """
        return -0.5*self._noise_log_likelihood
    
    
    def log_likelihood(self):
        """
        Calculates the real part of noise log-likelihood
        
        Returns
        -------
        float: The log likelihood value
        """
        strain_arr = self._construct_strain_arr()
        gw_proj_mat = self._calculate_gw_projection_matrix()
            
        Q = np.identity(self._totalNd) - gw_proj_mat
        null_nergy = np.sum( strain_arr * np.dot( Q, strain_arr) )
            
        return - 0.5*null_nergy
    
    
    
    
    
class StandardLikelihoodFDForSingleImage(bilby.Likelihood):
    def __init__(self, wf_dict, wf_dictL, trigTimeU, trigTimeL, deltaT):
        """
        A standard likelihood for single lensed image.

        Parameters
        ----------
        wf_dict: dict
            Dictonary of the unlensed event which in each members key is
            the detector name and corresponding value is a subdictonary.
            The members of each subdictonary contains the frequency samples and
            reconstructed whiten strain data in frequency-doamin.
            
        wf_dictL: dict
            Dictonary of the lensed image event which in each members key is
            the detector name and corresponding value is a subdictonary.
            The members of each subdictonary contains the frequency samples and
            reconstructed whiten strain data in frequency-domain.
            
        trigTimeU: float
            Trigger time of the event. 
        trigTimeL: float
            Trigger time of the lensed event.
             
        Init parameters
        ---------------
        mu: float
            The value of magnification factor.
        time: float
            GPS time of unlensed event in the geocentric frame.
        timeL: float
            GPS time of lensed event in the geocentric frame.
        psi: float
            Polarization angle of the unlensed event.
        psiL: float
            Polarization angle of the lensed event.
        ra: float 
            Right ascension of the source in radians
        dec: float
            Declination of the source in radians
        """
        super().__init__(parameters={'mu': None, 'time': None, 'timeL': None, 'psi': None, 'psiL': None, \
                           'phase': None, 'ra': None, 'dec': None})
        
    
        self._wf_dict = wf_dict
        self._wf_dictL = wf_dictL
        self._det_list = list( wf_dict.keys() )
        self._det_listL = list( wf_dictL.keys() )
    
        self._Nd = len( self._det_list )
        self._NdL = len( self._det_listL )
        self._totalNd = self._Nd + self._NdL
    
        self._timeU = trigTimeU
        self._timeL = trigTimeL
        self._sampling_rate = int(1.0/deltaT)
        
        self._noise_log_likelihood = np.sum( [np.sum(np.abs(item['strain'])**2) for det, item in wf_dict.items()] ) + \
                    np.sum( [ np.sum(np.abs(item['strain'])**2) for det, item in wf_dictL.items()] )
                                     
    
    
    def _calculate_gw_projection_matrix(self):
        """
        This function calculates a projection operator that projects the data
        into the subspace spanned by plus polarization (F+) and cros polarization (Fx).
        """
        
        mu = self.parameters['mu']
        time = self.parameters['time']
        timeL = self.parameters['timeL']
        ra = self.parameters['ra']
        dec = self.parameters['dec']
        psi = self.parameters['psi']
        psiL = self.parameters['psiL']
        
        mu = mu**0.5
        
        Amat = np.zeros(( self._totalNd, 2  ))
        for ii in range( self._Nd ):
            det = self._det_list[ii]
            detector = lal.cached_detector_by_prefix[ det ]
            Fp, Fc = lal.ComputeDetAMResponse(detector.response, ra, \
                                        dec, psi, lal.GreenwichMeanSiderealTime(time))
            Amat[ii] = [Fp, Fc]
        
        for jj in range( self._NdL ):
            det = self._det_listL[jj]
            detector = lal.cached_detector_by_prefix[ det ]
            Fp, Fc = lal.ComputeDetAMResponse(detector.response, ra, \
                                        dec, psiL, lal.GreenwichMeanSiderealTime(timeL))
            Amat[self._Nd+jj] = [mu*Fp, mu*Fc]
            
        Amat = np.matrix(Amat)
        P = np.matmul( Amat, np.matmul( np.matmul( Amat.T, Amat ).I, Amat.T))
        return np.array(P)
    
    
    def _construct_strain_arr(self):  
        """
        This function construct an totalNd x Ns array for the time-domain strains,
        where totalNd is the total number of strains and Ns is number of samples in
        a strain. Each strain array of the 2D array is time shifted according to the
        sky location and proposed time.
        """
        time = self.parameters['time']
        timeL = self.parameters['timeL']
        phase = self.parameters['phase']
        ra = self.parameters['ra']
        dec = self.parameters['dec']
        
        freqs = self._wf_dict[self._det_list[0]]['freqs']
        strain_arr = []        
        for det in self._det_list:
            t_delay = lal.TimeDelayFromEarthCenter( \
                            lalsim.DetectorPrefixToLALDetector( det ).location, ra, dec, time)
            total_tdelay = int((self._timeU - time + t_delay )*self._sampling_rate)/self._sampling_rate
            strain_arr.append(  self._wf_dict[det]['strain'] * \
                              np.exp(2.0*lal.PI*1j*freqs*total_tdelay) ) 
        
        for det in self._det_listL:
            t_delay = lal.TimeDelayFromEarthCenter( \
                            lalsim.DetectorPrefixToLALDetector( det ).location, ra, dec, timeL)
            total_tdelay = int((self._timeL - timeL + t_delay )*self._sampling_rate)/self._sampling_rate
            strain_arr.append(  self._wf_dictL[det]['strain']* \
                      np.exp(2.0*lal.PI*1j*freqs*total_tdelay + 1j*phase) ) 
            
        strain_arr = np.array(strain_arr)
        return strain_arr
    
    
    def noise_log_likelihood(self):
        """
        Calculates the noise log-likelihood
        
        Returns
        -------
        float: The noise log likelihood value
        """
        return - self._noise_log_likelihood
    
    
    def log_likelihood(self):
        """
        Calculates the log-likelihood
        
        Returns
        -------
        float: The log likelihood value
        """
        strain_arr = self._construct_strain_arr()
        gw_proj_mat = self._calculate_gw_projection_matrix()
            
        Q = np.identity(self._totalNd) - gw_proj_mat
        proj_strain_arr = np.dot( Q, strain_arr)
        null_nergy =  2.0*np.sum( strain_arr.conjugate() * proj_strain_arr).real
            
        return - 0.5*null_nergy
    
    
   
    
    