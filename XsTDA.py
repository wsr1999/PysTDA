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
        L_oo = self._get_L_oo()
        L_vv = self._get_L_vv()

        diag = self.eia.reshape(no, nv).copy()
        diag -= self.ax * np.einsum("uii,vaa,uv->ia", L_oo, L_vv, self.J, optimize=True)
        if self.do_RSH:
            diag -= self.betax * np.einsum(
                "uii,vaa,uv->ia", L_oo, L_vv, self.J_sr, optimize=True
            )
        if self.singlet:
            L_ov = self._get_L_ov()
            diag += 2.0 * np.einsum("uia,via,uv->ia", L_ov, L_ov, self.J, optimize=True)
        return diag.ravel()

    def matvec_A(self, x):
        mol = self.mol
        x = np.asarray(x).reshape(mol.nocc_trunc, mol.nvir_trunc)
        Ax = self.eia.reshape(mol.nocc_trunc, mol.nvir_trunc) * x

        L_oo = self._get_L_oo()
        L_vv = self._get_L_vv()
        if self.singlet:
            L_ov = self._get_L_ov()
            y = np.einsum("vjb,jb->v", L_ov, x, optimize=True)
            jy = self.J @ y
            Ax += 2.0 * np.einsum("uia,u->ia", L_ov, jy, optimize=True)

        m = np.einsum("vab,jb->vja", L_vv, x, optimize=True)
        jm = np.einsum("uv,vja->uja", self.J, m, optimize=True)
        Ax -= self.ax * np.einsum("uij,uja->ia", L_oo, jm, optimize=True)
        if self.do_RSH:
            jm_sr = np.einsum("uv,vja->uja", self.J_sr, m, optimize=True)
            Ax -= self.betax * np.einsum("uij,uja->ia", L_oo, jm_sr, optimize=True)
        return Ax.ravel()
