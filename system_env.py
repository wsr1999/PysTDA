# Copyright 2026 Sheng-Rui Wang
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from scipy import linalg

class molecule:
    def __init__(self, mo_coeff, mo_energy, mo_occ, aorange, atom_charge, atom_coord,
                 basis=None, nocc_trunc=None, nvir_trunc=None, int1e_ovlp=None,
                 cart=False):
        self.aorange = aorange
        self.atom_charge = atom_charge
        self.atom_coord = atom_coord
        self.basis = basis
        self.mo_coeff = mo_coeff
        self.mo_energy = mo_energy
        self.mo_occ = mo_occ
        self.nocc_trunc = nocc_trunc
        self.nvir_trunc = nvir_trunc
        self.int1e_ovlp = int1e_ovlp
        self.cart = cart

        self.natom = len(atom_charge)
        self.nao = mo_coeff.shape[0]
        self.occidx = np.where(mo_occ > 0)[0]
        self.viridx = np.where(mo_occ == 0)[0]
        self.nocc = len(self.occidx)
        self.nvir = len(self.viridx)
        self.e_homo = mo_energy[self.nocc-1]
        self.e_lumo = mo_energy[self.nocc]
        if nocc_trunc is None:
            nocc_trunc = len(self.occidx)
        if nvir_trunc is None:
            nvir_trunc = len(self.viridx)
        self.occidx_trunc = self.occidx[-nocc_trunc:]
        self.viridx_trunc = self.viridx[:nvir_trunc]
        self.X_ortho = self._get_X_ortho()
        self.C_MO = self._get_C_MO()

    def _get_X_ortho(self):
        r'''
        X = S^{-1/2} = U s^{-1/2} U^\dagger
        '''
        S = self.int1e_ovlp
        eigvals, eigvecs = linalg.eigh(S)
        self._overlap_eigvals = eigvals
        self._overlap_eigvecs = eigvecs
        X_ortho = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
        return X_ortho
    
    def _get_C_MO(self):
        r'''
        X = S^{-1/2} = U s^{-1/2} U^\dagger
        X^\dagger S X = I
        C' = X^{-1} C = S^{1/2} C
        Returns orthogonalized MO coefficients.
        '''
        eigvals = self._overlap_eigvals
        eigvecs = self._overlap_eigvecs
        return eigvecs @ (np.sqrt(eigvals)[:, None] * (eigvecs.T @ self.mo_coeff))

    def get_dist(self, dependence='atom'):
        r'''
        R_AB = |R_A - R_B|
        dependence: 'atom' or 'basis'
        returns: dist matrix
            If dependence='atom': shape (natom, natom)
            If dependence='basis': shape (nbasis, nbasis)
        '''
        coord = self.atom_coord
        if dependence == 'atom':
            # Vectorized distance calculation: (N, 1, 3) - (1, N, 3) -> (N, N, 3) -> norm -> (N, N)
            dist = np.linalg.norm(coord[:, None, :] - coord[None, :, :], axis=-1)
            return dist
        elif dependence == 'basis':
            ao_coords = []
            for A in range(self.natom):
                ao_start, ao_end = self.aorange[A, 2:4]
                ao_coords.append(np.tile(coord[A], (ao_end - ao_start, 1)))
            ao_coords = np.vstack(ao_coords)
            dist = np.linalg.norm(ao_coords[:, None, :] - ao_coords[None, :, :], axis=-1)
            return dist
        else:
            raise ValueError("Invalid dependence. Choose 'atom' or 'basis'.")
        
    def _charge_center(self):
        r'''
        Calculate the charge center of the molecule.
        R_c = \sum_A Z_A R_A / \sum_A Z_A
        '''
        return np.einsum('z,zr->r', self.atom_charge, self.atom_coord) / np.sum(self.atom_charge)


