import numpy as np


def spin_echo_equation(PD, T1, T2, TR, TE):
    """
    https://www.cis.rit.edu/htbooks/mri/chap-10/chap-10.htm
    """
    TR, TE = float(TR), float(TE)
    signal = PD * (1 - np.exp(-TR/T1)) * np.exp(-TE/T2)
    return signal


def flair_equation(PD, T1, T2, TR, TE, TI):
    """
    https://mriquestions.com/choice-of-ir-parameters.html
    """
    TR, TE, TI = float(TR), float(TE), float(TI)    
    signal = PD * (1 - 2*np.exp(-TI/T1) + np.exp(-TR/T1)) * np.exp(-TE/T2)
    return np.abs(signal)


def calc_flair_ti(TR, csf_t1):
    TI = csf_t1 * (np.log(2) - np.log(1 + np.exp(-TR/csf_t1))) # Null-point of CSF (https://mriquestions.com/ti-to-null-a-tissue.html)
    return TI


def ultsynth_contrast_equation(pd_vec, t1_vec, t2_vec, alpha, phi, beta_vec, TI, TE):
    """
    Simulation based on:
    - Adams et al. UltimateSynth: MRI Physics for Pan-Contrast AI. bioRXiv 2025 (https://www.biorxiv.org/content/10.1101/2024.12.05.627056v1)
    """
    alpha, phi = np.radians(alpha), np.radians(phi)
    TI, TE = float(TI), float(TE)
    mz_vec = pd_vec * (1 - 2*np.exp(-TI/t1_vec))
    m_vec = np.stack([np.zeros_like(mz_vec), np.zeros_like(mz_vec), mz_vec], axis=0)
    R = np.stack([
        np.stack([np.exp(-TE/t2_vec),    np.zeros_like(t2_vec), np.zeros_like(t2_vec)], axis=1),
        np.stack([np.zeros_like(t2_vec), np.exp(-TE/t2_vec),    np.zeros_like(t2_vec)], axis=1),
        np.stack([np.zeros_like(t2_vec), np.zeros_like(t2_vec), np.exp(-TE/t1_vec)   ], axis=1)], axis=0)
    Q1 = np.array([
        [np.cos(phi),  np.sin(phi), 0],
        [-np.sin(phi), np.cos(phi), 0],
        [0,            0,           1]])
    Q2 = np.stack([
        np.stack([np.ones_like(beta_vec),  np.zeros_like(beta_vec),   np.zeros_like(beta_vec) ], axis=1),
        np.stack([np.zeros_like(beta_vec), np.cos(beta_vec * alpha),  np.sin(beta_vec * alpha)], axis=1),
        np.stack([np.zeros_like(beta_vec), -np.sin(beta_vec * alpha), np.cos(beta_vec * alpha)], axis=1)], axis=0)
    Q3 = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi),  0],
        [0,           0,            1]])
    Q = np.dot(Q1, np.transpose(np.dot(Q2, Q3), axes=(0,2,1)))  # Q = Q1 Q2 Q3
    R = np.transpose(R, axes=(0,2,1))
    A = np.zeros((3, 3, m_vec.shape[1]))  # A = R Q
    for i in range(3):
        for j in range(3):
            A[i,j,:] = np.sum(R[i,:,:] * Q[:,j,:], axis=0)        
    B = np.stack([np.zeros_like(t1_vec), np.zeros_like(t1_vec), mz_vec * (1 - np.exp(-TE/t1_vec))], axis=0)
    signal_vec = np.zeros_like(m_vec)  # signal_vec = A m_vec + B
    for i in range(3):
        signal_vec[i,:] = np.sum(A[i,:,:] * m_vec, axis=0) + B[0]
    signal = signal_vec[0,:] + 1j * signal_vec[1,:]
    return signal