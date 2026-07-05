# Copyright 2026 Sheng-Rui Wang
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import global_param
import simplified_int as sint
from TDA_base import TDA_base


class sTDA(TDA_base):
    def __init__(self, mol, ax, alpha=-100.0, beta=-100.0, singlet=True, nroot=10):
        super().__init__(mol, singlet=singlet, nroot=nroot)
        self.ax = ax
        self.alpha = 1.42 + 0.48 * ax if alpha <= -99.0 else alpha
        self.beta = 0.20 + 1.83 * ax if beta <= -99.0 else beta

        hardness = global_param.hardness()
        self.eta = hardness[np.array(mol.atom_charge) - 1]
        self.gammaJ, self.gammaK = self._get_gamma()

    def _get_L_oo(self):
        if self.L_oo is None:
            self.L_oo = sint.get_lowdin_pop(self.mol, type="oo", dependence="atom")
        return self.L_oo

    def _get_L_vv(self):
        if self.L_vv is None:
            self.L_vv = sint.get_lowdin_pop(self.mol, type="vv", dependence="atom")
        return self.L_vv

    def _get_L_ov(self):
        if self.L_ov is None:
            self.L_ov = sint.get_lowdin_pop(self.mol, type="ov", dependence="atom")
        return self.L_ov

    def _get_gamma(self):
        dist = self.mol.get_dist(dependence="atom")
        gammaJ = sint.get_MNOK(dist, self.beta, self.ax * self.eta, dependence="atom")
        gammaK = sint.get_MNOK(dist, self.alpha, self.eta, dependence="atom")
        return gammaJ, gammaK

    def get_JK(self):
        L_oo = self._get_L_oo()
        L_vv = self._get_L_vv()
        L_ov = self._get_L_ov()
        ijab_J = np.einsum("Aij,Bab,AB->ijab", L_oo, L_vv, self.gammaJ, optimize=True)
        iajb_K = np.einsum("Aia,Bjb,AB->iajb", L_ov, L_ov, self.gammaK, optimize=True)
        return iajb_K, ijab_J

    def get_A_matrix(self):
        mol = self.mol
        A = np.zeros((self.nov, self.nov))
        A.ravel()[:: self.nov + 1] = self.eia
        A = A.reshape(mol.nocc_trunc, mol.nvir_trunc, mol.nocc_trunc, mol.nvir_trunc)

        L_oo = self._get_L_oo()
        L_vv = self._get_L_vv()
        ijab_J = np.einsum("Aij,Bab,AB->ijab", L_oo, L_vv, self.gammaJ, optimize=True)
        ijab_J = ijab_J.transpose(0, 2, 1, 3)
        A -= ijab_J

        if self.singlet:
            L_ov = self._get_L_ov()
            iajb_K = np.einsum("Aia,Bjb,AB->iajb", L_ov, L_ov, self.gammaK, optimize=True)
            A += 2.0 * iajb_K
        return A

    def get_diag(self):
        no = self.mol.nocc_trunc
        nv = self.mol.nvir_trunc
        L_oo = self._get_L_oo()
        L_vv = self._get_L_vv()

        diag = self.eia.reshape(no, nv).copy()
        diag -= np.einsum("Aii,Baa,AB->ia", L_oo, L_vv, self.gammaJ, optimize=True)
        if self.singlet:
            L_ov = self._get_L_ov()
            diag += 2.0 * np.einsum("Aia,Bia,AB->ia", L_ov, L_ov, self.gammaK, optimize=True)
        return diag.ravel()

    def matvec_A(self, x):
        mol = self.mol
        x = np.asarray(x).reshape(mol.nocc_trunc, mol.nvir_trunc)
        Ax = self.eia.reshape(mol.nocc_trunc, mol.nvir_trunc) * x

        if self.singlet:
            L_ov = self._get_L_ov()
            y = np.einsum("Bjb,jb->B", L_ov, x, optimize=True)
            gamma_y = self.gammaK @ y
            Ax += 2.0 * np.einsum("Aia,A->ia", L_ov, gamma_y, optimize=True)

        L_oo = self._get_L_oo()
        L_vv = self._get_L_vv()
        m = np.einsum("Bab,jb->Bja", L_vv, x, optimize=True)
        gamma_m = np.einsum("AB,Bja->Aja", self.gammaJ, m, optimize=True)
        Ax -= np.einsum("Aij,Aja->ia", L_oo, gamma_m, optimize=True)
        return Ax.ravel()

    def matmat_A(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            return self.matvec_A(X)
        if X.ndim != 2 or X.shape[0] != self.nov:
            raise ValueError(f"X must have shape ({self.nov}, nvec).")

        mol = self.mol
        no = mol.nocc_trunc
        nv = mol.nvir_trunc
        nvec = X.shape[1]
        zs = X.T.reshape(nvec, no, nv)
        eia = self.eia.reshape(no, nv)
        L_oo = self._get_L_oo()
        L_vv = self._get_L_vv()

        AX = np.einsum("xia,ia->xia", zs, eia, optimize=True)
        if self.singlet:
            L_ov = self._get_L_ov()
            y = np.einsum("Bjb,xjb->xB", L_ov, zs, optimize=True)
            gamma_y = np.einsum("AB,xB->xA", self.gammaK, y, optimize=True)
            AX += 2.0 * np.einsum("Aia,xA->xia", L_ov, gamma_y, optimize=True)

        m = np.einsum("Bab,xjb->xBja", L_vv, zs, optimize=True)
        gamma_m = np.einsum("AB,xBja->xAja", self.gammaJ, m, optimize=True)
        AX -= np.einsum("Aij,xAja->xia", L_oo, gamma_m, optimize=True)
        return AX.reshape(nvec, -1).T
