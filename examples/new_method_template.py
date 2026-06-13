# Copyright 2026 Sheng-Rui Wang
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from TDA_base import TDA_base


class NewSemiEmpiricalTDA(TDA_base):
    """Minimal template for a new semi-empirical TDA method."""

    def __init__(self, mol, parameter=1.0, singlet=True, nroot=10):
        super().__init__(mol, singlet=singlet, nroot=nroot)
        self.parameter = parameter

    def get_diag(self):
        return self.eia.copy()

    def get_A_matrix(self):
        A = np.zeros((self.nov, self.nov))
        A.ravel()[:: self.nov + 1] = self.get_diag()
        return A.reshape(
            self.mol.nocc_trunc,
            self.mol.nvir_trunc,
            self.mol.nocc_trunc,
            self.mol.nvir_trunc,
        )

    def matvec_A(self, x):
        return self.get_diag() * np.asarray(x)
