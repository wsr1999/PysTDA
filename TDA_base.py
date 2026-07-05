# Copyright 2026 Sheng-Rui Wang
# SPDX-License-Identifier: Apache-2.0

import sys

import numpy as np
from scipy import linalg

import simplified_int as sint


def _load_pyscf_lib():
    try:
        from pyscf import lib
    except ImportError as exc:
        raise ImportError(
            "kernel_davidson requires the optional PySCF dependency. "
            "Install pyscf before using the iterative Davidson solver."
        ) from exc
    return lib


def _load_pyscf_rhf_tda():
    try:
        from pyscf.tdscf import rhf
    except ImportError as exc:
        raise ImportError(
            "analyze requires the optional PySCF dependency. "
            "Install pyscf before using the PySCF-backed analyzer."
        ) from exc
    return rhf.TDA


class _FakePyscfMF:
    def __init__(self, mol, stda_mol, source_mf=None):
        self.mol = mol
        self.mo_coeff = np.array(stda_mol.mo_coeff, copy=True)
        self.mo_occ = np.array(stda_mol.mo_occ, copy=True)
        self.mo_energy = np.array(stda_mol.mo_energy, copy=True)
        self.verbose = getattr(source_mf, "verbose", getattr(mol, "verbose", 5))
        self.stdout = getattr(source_mf, "stdout", sys.stdout)
        self.max_memory = getattr(source_mf, "max_memory", 4000)
        self.chkfile = getattr(source_mf, "chkfile", None)
        self.converged = getattr(source_mf, "converged", True)
        self.e_tot = getattr(source_mf, "e_tot", 0.0)

    def run(self):
        return self

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        return self


class TDA_base:
    def __init__(self, mol, singlet=True, nroot=10):
        self.mol = mol
        self.singlet = bool(singlet)
        self.nroot = int(nroot)
        self.nov = mol.nocc_trunc * mol.nvir_trunc

        C_MO = mol.C_MO
        self.C_occ = C_MO[:, mol.occidx_trunc]
        self.C_vir = C_MO[:, mol.viridx_trunc]
        self.eia = self._get_eia()

        if self.nroot > self.nov:
            print(f"Warning: nroot={self.nroot} exceeds the nov={self.nov} size "
                "of the truncated space. Setting nroot=nov.")
            self.nroot = self.nov

        self.e = None
        self.X = None
        self.X_full = None
        self.f = None
        self.L_oo = None
        self.L_vv = None
        self.L_ov = None

    def _get_L_oo(self):
        raise NotImplementedError("_get_L_oo must be implemented in a subclass.")

    def _get_L_vv(self):
        raise NotImplementedError("_get_L_vv must be implemented in a subclass.")

    def _get_L_ov(self):
        raise NotImplementedError("_get_L_ov must be implemented in a subclass.")

    def _get_eia(self):
        mol = self.mol
        e_occ = mol.mo_energy[mol.occidx_trunc]
        e_vir = mol.mo_energy[mol.viridx_trunc]
        return (e_vir[None, :] - e_occ[:, None]).ravel()

    def get_A_matrix(self):
        raise NotImplementedError("get_A_matrix must be implemented in a subclass.")

    def matvec_A(self, x):
        raise NotImplementedError("matvec_A must be implemented in a subclass.")

    def matmat_A(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            return self.matvec_A(X)
        if X.ndim != 2 or X.shape[0] != self.nov:
            raise ValueError(f"X must have shape ({self.nov}, nvec).")
        if X.shape[1] == 0:
            return np.empty((self.nov, 0), dtype=X.dtype)
        return np.column_stack([self.matvec_A(X[:, i]) for i in range(X.shape[1])])

    def get_diag(self):
        return self.eia.copy()

    def kernel_diag(self):
        A_matrix = self.get_A_matrix().reshape(self.nov, self.nov)
        kwargs = {"check_finite": False, "overwrite_a": True}
        if self.nroot < self.nov:
            kwargs["subset_by_index"] = [0, self.nroot - 1]
        eigvals, eigvecs = linalg.eigh(A_matrix, **kwargs)

        self._store_solution(eigvals[:self.nroot], eigvecs[:, :self.nroot])
        return self.e, self.X_full

    def kernel_davidson(self, max_cycle=100, tol=1e-6, max_space=20, max_memory=4000, verbose=2):
        lib = _load_pyscf_lib()
        x0 = self._init_guess()
        diag = np.array(self.get_diag(), dtype=float, copy=True).ravel()
        if diag.shape != (self.nov,):
            raise ValueError(f"get_diag must return a vector with shape ({self.nov},).")
        precond = lib.make_diag_precond(diag)
        max_space = max(int(max_space), self.nroot + 2)

        def aop(xs):
            if isinstance(xs, np.ndarray):
                if xs.ndim == 1:
                    X = xs.reshape(self.nov, 1)
                elif xs.ndim == 2 and xs.shape[1] == self.nov:
                    X = xs.T
                elif xs.ndim == 2 and xs.shape[0] == self.nov:
                    X = xs
                else:
                    raise ValueError(
                        "Davidson trial vectors must have shape "
                        f"({self.nov}, nvec) or (nvec, {self.nov})."
                    )
            else:
                xs = list(xs)
                if len(xs) == 0:
                    return []
                X = np.column_stack([np.asarray(x).ravel() for x in xs])
                if X.shape[0] != self.nov:
                    raise ValueError(f"Davidson trial vectors must have length {self.nov}.")

            AX = np.asarray(self.matmat_A(X))
            if AX.shape != X.shape:
                raise ValueError(f"matmat_A must return shape {X.shape}, got {AX.shape}.")
            return [AX[:, i].copy() for i in range(AX.shape[1])]

        conv, eigvals, eigvecs = lib.davidson1(aop, x0, precond, tol=tol, max_cycle=max_cycle,
                                               max_space=max_space, max_memory=max_memory, 
                                               nroots=self.nroot, verbose=verbose)
        if not np.all(np.asarray(conv, dtype=bool)):
            raise RuntimeError("PySCF Davidson solver did not converge.")

        eigvals = np.atleast_1d(np.asarray(eigvals, dtype=float))
        eigvecs = self._as_column_vectors(eigvecs)
        order = np.argsort(eigvals)[:self.nroot]
        self._store_solution(eigvals[order], eigvecs[:, order])
        return self.e, self.X_full

    def _init_guess(self):
        guess_idx = np.argsort(self.eia)[:self.nroot]
        guesses = []
        for idx in guess_idx:
            x = np.zeros(self.nov)
            x[idx] = 1.0
            guesses.append(x)
        return guesses

    def _as_column_vectors(self, eigvecs):
        eigvecs = np.asarray(eigvecs)
        if eigvecs.ndim == 1:
            eigvecs = eigvecs[None, :]
        if eigvecs.shape[0] == self.nroot and eigvecs.shape[1] == self.nov:
            return eigvecs.T
        if eigvecs.shape[0] == self.nov:
            return eigvecs
        raise ValueError("Davidson eigenvectors have an unexpected shape.")

    def _store_solution(self, eigvals, eigvecs):
        nroot = len(eigvals)
        self.e = np.array(eigvals, copy=True)
        self.X = np.array(eigvecs[:, :nroot], copy=True) * np.sqrt(0.5)
        self.X_full = (self.get_full_X().reshape(self.mol.nocc, self.mol.nvir, nroot).transpose(2, 0, 1))

    def get_full_X(self):
        if self.X is None:
            raise RuntimeError("No excitation vectors are available. Run kernel_diag first.")

        mol = self.mol
        nroot = self.X.shape[1]
        X_full = np.zeros((nroot, mol.nocc, mol.nvir), dtype=self.X.dtype)
        occ_rows = mol.occidx_trunc - mol.occidx[0]
        vir_cols = mol.viridx_trunc - mol.viridx[0]
        X_full[:, occ_rows[:, None], vir_cols[None, :]] = (self.X.T.reshape(nroot, mol.nocc_trunc, mol.nvir_trunc))
        return X_full.transpose(1, 2, 0).reshape(mol.nocc * mol.nvir, nroot)

    def analyze(self, mf=None, save_v=False, v_filename="eig_vec.dat",
                print_g=False, verbose=None):
        if self.e is None or self.X_full is None:
            raise RuntimeError(
                "No excitation solution is available. Run kernel_diag or "
                "kernel_davidson before analyze."
            )

        pyscf_tda_cls = _load_pyscf_rhf_tda()
        pyscf_mol = self._get_analyzer_mol(mf)
        fake_mf = _FakePyscfMF(pyscf_mol, self.mol, source_mf=mf)
        tdobj = pyscf_tda_cls(fake_mf)
        nstates = len(self.e)

        tdobj.nstates = nstates
        tdobj.singlet = self.singlet
        tdobj.e = np.array(self.e, copy=True)
        tdobj.xy = [
            (np.array(x, copy=True), np.zeros_like(x))
            for x in self.X_full[:nstates]
        ]
        tdobj.converged = np.ones(nstates, dtype=bool)

        if save_v:
            np.savetxt(
                v_filename,
                self.X_full[:nstates].reshape(nstates, -1),
                header="pysTDA full occupied-virtual amplitudes; rows are states",
            )

        tdobj.analyze(verbose=verbose)
        if print_g:
            self._print_g_values(tdobj)
        return tdobj

    def _get_analyzer_mol(self, mf):
        if mf is not None:
            return mf.mol
        if getattr(self.mol, "basis", None) is None:
            raise ValueError(
                "PySCF molecule reconstruction requires mol.basis. Pass a PySCF "
                "mean-field object as analyze(mf=mf) for systems without basis metadata."
            )
        return sint.pyscf_build_mol(self.mol, cart=getattr(self.mol, "cart", False))

    def _print_g_values(self, tdobj):
        try:
            trans_dip = tdobj.transition_dipole()
            trans_m = tdobj.transition_magnetic_dipole()
        except NotImplementedError as exc:
            raise RuntimeError(
                "PySCF cannot compute magnetic-dipole g values for this system, "
                "for example with ECP or pseudopotential references."
            ) from exc

        print("State   absmu(10^-18 esu cm)   absm(10^-20 erg/G)   "
              "costheta         g              g*1000")
        print("-" * 90)
        for i in range(len(self.e)):
            dip = np.asarray(trans_dip[i]).real
            mag = np.asarray(trans_m[i]).real
            absmu_au = np.linalg.norm(dip)
            absmu = absmu_au / 0.393456
            absm_au = np.linalg.norm(mag)
            absm = absm_au * 1.8548
            if absmu_au < 1.0e-12 or absm_au < 1.0e-12:
                costheta = 0.0
                g = 0.0
            else:
                costheta = float(np.dot(dip, mag) / (absmu_au * absm_au))
                g = 4.0 * costheta * absm * 1.0e-20 / (absmu * 1.0e-18)
            print(f"{i+1:5d}   {absmu:20.6f}   {absm:18.6f}   "
                  f"{costheta:12.6f}   {g:12.6f}   {g*1000:12.6f}")

    def oscillator_strength(self, dipole_int, gauge="length"):
        # Formula and control flow adapted from PySCF TD-SCF utilities and
        # modified for pysTDA's stored TDA amplitudes.
        if self.e is None or self.X_full is None:
            raise RuntimeError("No excitation solution is available. Run kernel_diag first.")

        if gauge == "length":
            trans_dip = self.contract_multipole(dipole_int)
            f = 2.0 / 3.0 * np.einsum("s,sx,sx->s", self.e, trans_dip, trans_dip, optimize=True)
        elif gauge == "velocity":
            trans_dip = -self.contract_multipole(dipole_int)
            f = 2.0 / 3.0 * np.einsum("s,sx,sx->s", 1.0 / self.e, trans_dip, trans_dip, optimize=True)
        else:
            raise ValueError("Gauge must be 'length' or 'velocity'.")

        self.f = f if self.singlet else np.zeros_like(self.e)
        return self.f

    def contract_multipole(self, dipole_int):
        # Multipole contraction structure adapted from PySCF TD-SCF utilities
        # and modified for pysTDA's full occupied-virtual amplitude layout.
        if self.X_full is None:
            raise RuntimeError("No excitation vectors are available. Run kernel_diag first.")

        mol = self.mol
        pol_shape = dipole_int.shape[:-2]
        dipole_int = dipole_int.reshape(-1, mol.nao, mol.nao)
        mo_occ = mol.mo_coeff[:, mol.occidx]
        mo_vir = mol.mo_coeff[:, mol.viridx]

        ints = np.einsum("mi,xmn,na->xia", mo_occ, dipole_int, mo_vir.conj(), optimize=True)
        pol = 2.0 * np.einsum("sia,xia->sx", self.X_full, ints, optimize=True)
        return pol.reshape((self.X_full.shape[0],) + pol_shape)
