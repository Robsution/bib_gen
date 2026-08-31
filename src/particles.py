"""Particle-name helpers used in analysis plots."""

PDG_LABELS = {
    22: "γ",
    11: "e⁻",
    -11: "e⁺",
    2112: "n",
    2212: "p",
    -2212: "p̄",
    13: "μ⁻",
    -13: "μ⁺",
    111: "π⁰",
    211: "π⁺",
    -211: "π⁻",
    130: "K⁰L",
    310: "K⁰S",
    321: "K⁺",
    -321: "K⁻",
}


def particle_label(pdg_code: int) -> str:
    return PDG_LABELS.get(int(pdg_code), str(int(pdg_code)))
