# Copyright 2026 Sheng-Rui Wang
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import simplified_int as sint
from TDA_base import TDA_base


class XsTDA(TDA_base):
    def __init__(self, mol, ax, do_RSH=False, betax=None, omega=None, singlet=True, nroot=10):
        super().__init__(mol, singlet=singlet, nroot=nroot)
        self.ax = ax
        self.do_RSH = bool(do_RSH)
        self.betax = betax
        self.omega = omega

        if self.do_RSH and self.betax is None:
            raise ValueError("betax is required when do_RSH = True.")

        if self.do_RSH:
            self.J, self.J_sr = self._get_J_AO()
        else:
            self.J = self._get_J_AO()

    def _get_L_oo(self):
        if self.L_oo is None:
            self.L_oo = sint.get_lowdin_pop(self.mol, type="oo", dependence="basis")
        return self.L_oo

    def _get_L_vv(self):
        if self.L_vv is None:
            self.L_vv = sint.get_lowdin_pop(self.mol, type="vv", dependence="basis")
        return self.L_vv

    def _get_L_ov(self):
        if self.L_ov is None:
            self.L_ov = sint.get_lowdin_pop(self.mol, type="ov", dependence="basis")
        return self.L_ov

    def _get_J_AO(self):
        J = sint.get_uuvv_pyscf(self.mol, cart=self.mol.cart, do_RSH=False)
        if self.do_RSH:
            J_sr = sint.get_uuvv_pyscf(self.mol, cart=self.mol.cart, do_RSH=True, omega=self.omega)
            return J, J_sr
        return J

    def get_JK(self):
        L_oo = self._get_L_oo()
        L_vv = self._get_L_vv()
        L_ov = self._get_L_ov()

        ijab_J = np.einsum("uij,vab,uv->ijab", L_oo, L_vv, self.J, optimize=True)
        iajb_K = np.einsum("uia,vjb,uv->iajb", L_ov, L_ov, self.J, optimize=True)

        if self.do_RSH:
            ijab_J_sr = np.einsum("uij,vab,uv->ijab", L_oo, L_vv, self.J_sr, optimize=True)
            return iajb_K, ijab_J, ijab_J_sr
        return iajb_K, ijab_J

    def get_A_matrix(self):
        mol = self.mol
        A = np.zeros((self.nov, self.nov))
        A.ravel()[:: self.nov + 1] = self.eia
        A = A.reshape(mol.nocc_trunc, mol.nvir_trunc, mol.nocc_trunc, mol.nvir_trunc)

        L_oo = self._get_L_oo()
        L_vv = self._get_L_vv()
        ijab_J = np.einsum("uij,vab,uv->ijab", L_oo, L_vv, self.J, optimize=True)
        A -= self.ax * ijab_J.transpose(0, 2, 1, 3)
        if self.do_RSH:
            ijab_J_sr = np.einsum("uij,vab,uv->ijab", L_oo, L_vv, self.J_sr, optimize=True)
            A -= self.betax * ijab_J_sr.transpose(0, 2, 1, 3)

        if self.singlet:
            L_ov = self._get_L_ov()
            iajb_K = np.einsum("uia,vjb,uv->iajb", L_ov, L_ov, self.J, optimize=True)
            A += 2.0 * iajb_K
        return A

    def get_diag(self):
        no = self.mol.nocc_trunc
        nv = self.mol.nvir_trunc

        diag = self.eia.reshape(no, nv).copy()
        C_occ_sq = self.C_occ * self.C_occ
        C_vir_sq = self.C_vir * self.C_vir

        diag -= self.ax * (C_occ_sq.T @ (self.J @ C_vir_sq))
        if self.do_RSH:
            diag -= self.betax * (C_occ_sq.T @ (self.J_sr @ C_vir_sq))
        if self.singlet:
            diag += 2.0 * self._get_exchange_diag()
        return diag.ravel()

    def _get_exchange_diag(self):
        no = self.mol.nocc_trunc
        nv = self.mol.nvir_trunc
        diag = np.empty((no, nv), dtype=np.result_type(self.C_occ, self.C_vir, self.J))

        if no <= nv:
            for i in range(no):
                weighted_vir = self.C_vir * self.C_occ[:, i:i + 1]
                j_weighted_vir = self.J @ weighted_vir
                diag[i, :] = np.einsum("ua,ua->a", weighted_vir, j_weighted_vir, optimize=True)
        else:
            for a in range(nv):
                weighted_occ = self.C_occ * self.C_vir[:, a:a + 1]
                j_weighted_occ = self.J @ weighted_occ
                diag[:, a] = np.einsum("ui,ui->i", weighted_occ, j_weighted_occ, optimize=True)
        return diag

    def matvec_A(self, x):
        mol = self.mol
        x = np.asarray(x).reshape(mol.nocc_trunc, mol.nvir_trunc)
        Ax = self.eia.reshape(mol.nocc_trunc, mol.nvir_trunc) * x

        if self.singlet:
            y = np.einsum("ua,ia->ui", self.C_vir, x, optimize=True)
            z = np.einsum("ui,ui->u", self.C_occ, y, optimize=True)
            jz = self.J @ z
            Ax += 2.0 * np.einsum(
                "ui,u,ua->ia", self.C_occ, jz, self.C_vir, optimize=True
            )

        q = np.einsum("ui,ia,va->uv", self.C_occ, x, self.C_vir, optimize=True)
        tj = np.einsum("uv,uv,va->ua", self.J, q, self.C_vir, optimize=True)
        Ax -= self.ax * np.einsum("ui,ua->ia", self.C_occ, tj, optimize=True)
        if self.do_RSH:
            tsr = np.einsum("uv,uv,va->ua", self.J_sr, q, self.C_vir, optimize=True)
            Ax -= self.betax * np.einsum("ui,ua->ia", self.C_occ, tsr, optimize=True)
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

        AX = np.einsum("xia,ia->xia", zs, eia, optimize=True)
        if self.singlet:
            y = np.einsum("ua,xia->xui", self.C_vir, zs, optimize=True)
            z = np.einsum("ui,xui->xu", self.C_occ, y, optimize=True)
            jz = np.einsum("uv,xv->xu", self.J, z, optimize=True)
            AX += 2.0 * np.einsum(
                "ui,xu,ua->xia", self.C_occ, jz, self.C_vir, optimize=True
            )

        q = np.einsum("ui,xia,va->xuv", self.C_occ, zs, self.C_vir, optimize=True)
        tj = np.einsum("uv,xuv,va->xua", self.J, q, self.C_vir, optimize=True)
        AX -= self.ax * np.einsum("ui,xua->xia", self.C_occ, tj, optimize=True)
        if self.do_RSH:
            tsr = np.einsum("uv,xuv,va->xua", self.J_sr, q, self.C_vir, optimize=True)
            AX -= self.betax * np.einsum("ui,xua->xia", self.C_occ, tsr, optimize=True)
        return AX.reshape(nvec, -1).T
