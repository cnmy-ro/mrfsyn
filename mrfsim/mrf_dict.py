import numpy as np
import torch
from tqdm import tqdm
import h5py

from mrfsim.epg import *



class MRFDictionary:

    def __init__(self, seq, param_lists, device):

        self.seq = seq
        self.param_lists = param_lists  # {'t1': Tensor, 't2': Tensor, ...}
        self.device = device
        
        self.epg = EPG(seq, device)  # EPG engine
        self.fingerprints = None  # Dictionary of fingerprints
        self.compressed_fingerprints = None
        self.compression_matrix = None
        self.fingerprint_norm_factors = None
        self.compressed_fingerprint_norm_factors = None

    @torch.no_grad()
    def compute(self, batch_size=512):
    
        num_tissue_types = self.param_lists['t1'].shape[0]
        signal_length = len(self.seq.fa_pattern)
        self.fingerprints = torch.empty((num_tissue_types, signal_length), device='cpu', dtype=torch.cfloat)
        self.fingerprint_norm_factors = torch.empty((num_tissue_types, 1), device='cpu', dtype=torch.float)
        num_batches = num_tissue_types // batch_size
        num_batches += 1 if int(num_tissue_types % batch_size) > 0 else 0

        print("Computing dictionary ...")
        for b in tqdm(range(num_batches)):
            start, end = b * batch_size, min((b + 1) * batch_size, num_tissue_types)
            t1_list = self.param_lists['t1'][start : end].to(self.device)
            t2_list = self.param_lists['t2'][start : end].to(self.device)
            self.epg.simulate(t1_list=t1_list, t2_list=t2_list)
            echoes_batch = self.epg.find_echoes().cpu()
            self.epg.reset()
            echoes_norm_factors = torch.norm(echoes_batch, dim=1, keepdim=True, p=2)
            self.fingerprints[start : end, :] = echoes_batch / torch.clamp(echoes_norm_factors, 1e-10)
            self.fingerprint_norm_factors[start : end] = echoes_norm_factors
        if self.device == 'cuda': torch.cuda.empty_cache()
        print("Done")

    @torch.no_grad()
    def svd_compress(self, num_coeffs):
        print("Computing SVD ...")
        fingerprints = self.fingerprints * self.fingerprint_norm_factors
        U, S, Vh = torch.linalg.svd(fingerprints, full_matrices=False)
        V = Vh.mH
        self.compression_matrix = V[:, 0 : num_coeffs].clone()
        self.compressed_fingerprints = torch.mm(fingerprints, self.compression_matrix)
        self.compressed_fingerprint_norm_factors = torch.norm(self.compressed_fingerprints, dim=1, keepdim=True, p=2)
        self.compressed_fingerprints = self.compressed_fingerprints / torch.clamp(self.compressed_fingerprint_norm_factors, 1e-10)
        print("Done")

    @torch.no_grad()
    def match_signals(self, signals, compress_signals=False, batch_size=512):
        """
        Parameters:
            signals: Tensor of size (batch, length)
            compress_signals: Bool
        Returns:
            best_params: Dict[Tensor], each tensor of size (batch,)
        """
        num_tissue_types = self.param_lists['t1'].shape[0]
        num_signals = signals.shape[0]        
        match_scores = torch.empty((num_tissue_types, num_signals), device='cpu', dtype=torch.float)        

        # Compress, if needed
        if compress_signals:
            assert self.compression_matrix is not None
            assert signals.shape[1] == self.compression_matrix.shape[0]
            signals = torch.mm(signals, self.compression_matrix)
            fingerprints = self.compressed_fingerprints
            fingerprint_norm_factors = self.compressed_fingerprint_norm_factors
        else:
            if self.compression_matrix is not None and signals.shape[1] == self.compression_matrix.shape[1]:
                fingerprints = self.compressed_fingerprints
                fingerprint_norm_factors = self.compressed_fingerprint_norm_factors
            else:
                assert signals.shape[1] == self.fingerprints.shape[1]
                fingerprints = self.fingerprints
                fingerprint_norm_factors = self.fingerprint_norm_factors
        
        # Normalize signals
        signal_norm = torch.norm(signals, dim=1, keepdim=True, p=2)
        signals = signals / torch.clamp(signal_norm, 1e-10)
        signals = signals.to(self.device)
        
        # Do dict matching in batches
        print("Matching ...")
        num_batches = num_tissue_types // batch_size
        num_batches += 1 if int(num_tissue_types % batch_size) > 0 else 0        
        for b in tqdm(range(num_batches)):
            start, end = b * batch_size, min((b + 1) * batch_size, num_tissue_types)
            fingerprints_batch = fingerprints[start : end, :].to(self.device)
            match_scores[start : end, :] = torch.real(torch.mm(fingerprints_batch, signals.mH)).cpu()
        # Find best-matching params
        best_matches_idxs = torch.argmax(match_scores, dim=0)
        best_params = {param_name: param_list[best_matches_idxs] for param_name, param_list in self.param_lists.items()}
        best_params.update({'m0': (signal_norm / fingerprint_norm_factors[best_matches_idxs]).squeeze()})
        if self.device == 'cuda': torch.cuda.empty_cache()
        print("Done")

        # Return
        best_params = {param_name: param_list.cpu() for param_name, param_list in best_params.items()}
        return best_params

    @torch.no_grad()
    def match_image(self, mrf_image, compress_image=False):
        """
        Parameters:
            mrf_image: Tensor of size (height, width, mrf_channels)
            compress: Bool
        Returns:
            param_maps: Dict[Tensor], each tensor of size (height, width)
        """
        height, width, mrf_channels = mrf_image.shape[0], mrf_image.shape[1], mrf_image.shape[2]
        signals = mrf_image.reshape(height*width, mrf_channels)
        best_params = self.match_signals(signals, compress_image)
        param_maps = {param_name: param_list.reshape(height, width, mrf_channels) for param_name, param_list in best_params.items()}
        return param_maps

    def save(self, path):

        with h5py.File(path, 'w') as hf:

            _ = hf.create_dataset(name='fingerprints', data=self.fingerprints.numpy().astype(np.complex64))            
            _ = hf.create_dataset(name='fingerprint_norm_factors', data=self.fingerprint_norm_factors.numpy().astype(np.float32))
            if self.compression_matrix is not None:
                _ = hf.create_dataset(name='compressed_fingerprints', data=self.compressed_fingerprints.numpy().astype(np.complex64))
                _ = hf.create_dataset(name='compressed_fingerprint_norm_factors', data=self.compressed_fingerprint_norm_factors)
                _ = hf.create_dataset(name='compression_matrix', data=self.compression_matrix.numpy().astype(np.complex64))

            groups = {'param_lists': hf.create_group(name='param_lists')}
            for param_name in self.param_lists.keys():
                param_values = self.param_lists[param_name].numpy().astype(np.float32)
                _ = groups['param_lists'].create_dataset(name=param_name, data=param_values)

    def load(self, path):
        with h5py.File(path, 'r') as hf:
            self.param_lists = {k: torch.tensor(np.asarray(hf['param_lists'][k], dtype=np.float32)) for k in hf['param_lists'].keys()}
            self.fingerprints = torch.tensor(np.asarray(hf['fingerprints'], dtype=np.complex64))            
            self.fingerprint_norm_factors = torch.tensor(np.asarray(hf['fingerprint_norm_factors'], dtype=np.float32))
            if 'compression_matrix' in hf.keys():
                self.compressed_fingerprints = torch.tensor(np.asarray(hf['compressed_fingerprints'], dtype=np.complex64))
                self.compressed_fingerprint_norm_factors = torch.tensor(np.asarray(hf['compressed_fingerprint_norm_factors'], dtype=np.float32))
                self.compression_matrix = torch.tensor(np.asarray(hf['compression_matrix'], dtype=np.complex64))