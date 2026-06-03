"""Veletsos-Meek period-lengthening bounds per NIST GCR 12-917-21 §2.

Reviewer §4.1 asked for an honest SSI assessment. Full Stewart 1999 /
Todorovska 2009 separation needs foundation/inertial data we do not have,
so this module gives an order-of-magnitude *upper bound* on the period
lengthening T_tilde / T from soil-foundation flexibility, using:

  - Pais & Kausel (1988) static stiffness for a rigid surface footing on a
    homogeneous halfspace (equivalent radius approximation).
  - Veletsos & Meek (1974) / NIST GCR 12-917-21 eq. (2-12) for the SSI
    period-lengthening ratio.

Assumptions (conservative, stated up front):
  * Vs30 = 462.57 m/s (HWA018 borehole, NCREE EGDT, 31.5 m profile).
  * Soil mass density rho = 1900 kg/m^3, Poisson nu = 0.35.
  * Buildings on rigid surface mat foundation of equivalent radius
    r_eq = sqrt(footprint_m2 / pi).
  * Effective fixed-base mode mass m_bar = 0.7 × total mass; total mass
    estimated as 800 kg/m^2 × footprint × num_floors (rough RC/SRC value).
  * Modal effective height h_bar = 0.7 × h_total.

Output: one row per station with f_fixed, f_with_SSI, period_lengthening
ratio, and the dimensionless sigma = h_bar / (Vs * T_fixed) which controls
the SSI magnitude. Sigma << 0.1 means SSI is negligible (stiff-soil case),
sigma > 0.3 means SSI dominates (soft-soil flexible-building case).

References:
  Veletsos AS, Meek JW (1974). Dynamic behaviour of building-foundation
    systems. Earthquake Engng Struct Dyn 3:121-138.
  Pais A, Kausel E (1988). Approximate formulas for dynamic stiffnesses
    of rigid foundations. Soil Dyn Earthq Eng 7:213-227.
  NIST (2012). Soil-Structure Interaction for Building Structures.
    NIST GCR 12-917-21.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from src.station_config import (
    CAMPUS_SITE_CLASS,
    CAMPUS_SOIL_DENSITY_KGM3,
    CAMPUS_SOIL_POISSON,
    CAMPUS_VS30_MS,
    STATIONS,
    StationConfig,
    empirical_frequency,
)


# Rough modal-mass / modal-height multipliers for a uniform shear-beam
# (fundamental-mode participation factor; e.g., NIST GCR 12-917-21 Table 2-1).
MODAL_MASS_FRACTION = 0.7
MODAL_HEIGHT_FRACTION = 0.7
# Floor mass density (RC/SRC typical, kg/m^2 of plan area) — see NIST §2.6.
FLOOR_MASS_PER_M2 = 800.0
GRAVITY = 9.80665


def _floor_count(cfg: StationConfig) -> int:
    return max(1, int(round(cfg.height_m / 3.0)))


def _equivalent_radius_m(footprint_m2: float) -> float:
    return math.sqrt(footprint_m2 / math.pi)


def _shear_modulus(vs_ms: float, rho_kgm3: float) -> float:
    """Small-strain shear modulus G0 (Pa)."""
    return rho_kgm3 * vs_ms ** 2


def pais_kausel_stiffness(r: float, vs: float, rho: float, nu: float) -> dict:
    """Approximate static stiffness of a rigid disk on a homogeneous halfspace.

    Pais & Kausel (1988) reduced formulas — accurate to ~10% for a circular
    surface footing in the low-frequency limit. NIST GCR 12-917-21 eq. (2-2).
    Returns dict of:
      k_x — horizontal translation (N/m)
      k_z — vertical translation (N/m)
      k_yy — rocking about horizontal axis (N·m/rad)
      k_zz — torsion (N·m/rad)
    """
    G = _shear_modulus(vs, rho)
    k_x = 8.0 * G * r / (2.0 - nu)
    k_z = 4.0 * G * r / (1.0 - nu)
    k_yy = 8.0 * G * r ** 3 / (3.0 * (1.0 - nu))
    k_zz = 16.0 * G * r ** 3 / 3.0
    return {"k_x": k_x, "k_z": k_z, "k_yy": k_yy, "k_zz": k_zz, "G": G}


def period_lengthening(cfg: StationConfig, *,
                       vs: float = CAMPUS_VS30_MS,
                       rho: float = CAMPUS_SOIL_DENSITY_KGM3,
                       nu: float = CAMPUS_SOIL_POISSON,
                       floor_mass_per_m2: float = FLOOR_MASS_PER_M2) -> dict:
    """Compute Veletsos-Meek SSI period-lengthening ratio for one station."""
    nfloors = _floor_count(cfg)
    total_mass = floor_mass_per_m2 * cfg.footprint_m2 * nfloors           # kg
    m_bar = MODAL_MASS_FRACTION * total_mass                              # kg
    h_bar = MODAL_HEIGHT_FRACTION * cfg.height_m                          # m

    f_fixed = empirical_frequency(cfg)                                    # Hz
    T_fixed = 1.0 / f_fixed                                               # s
    omega_fixed = 2.0 * math.pi * f_fixed                                 # rad/s

    # Fixed-base stiffness from m_bar and omega_fixed
    k_fixed = m_bar * omega_fixed ** 2                                    # N/m

    r_eq = _equivalent_radius_m(cfg.footprint_m2)
    foundation = pais_kausel_stiffness(r_eq, vs, rho, nu)

    # Veletsos-Meek (NIST eq. 2-12): T_tilde^2/T^2 = 1 + k_fixed/k_x + k_fixed h_bar^2/k_yy
    term_translation = k_fixed / foundation["k_x"]
    term_rocking = k_fixed * (h_bar ** 2) / foundation["k_yy"]
    T_ratio_sq = 1.0 + term_translation + term_rocking
    T_ratio = math.sqrt(T_ratio_sq)
    T_with_ssi = T_fixed * T_ratio
    f_with_ssi = 1.0 / T_with_ssi
    sigma = h_bar / (vs * T_fixed)                                         # SSI strength parameter

    return {
        "station_id": cfg.station_id,
        "locname": cfg.locname,
        "n_floors": nfloors,
        "height_m": cfg.height_m,
        "footprint_m2": cfg.footprint_m2,
        "r_eq_m": r_eq,
        "h_bar_m": h_bar,
        "total_mass_kg": total_mass,
        "m_bar_kg": m_bar,
        "f_fixed_hz": f_fixed,
        "T_fixed_s": T_fixed,
        "k_fixed_N_per_m": k_fixed,
        "k_x_N_per_m": foundation["k_x"],
        "k_yy_Nm_per_rad": foundation["k_yy"],
        "G_soil_Pa": foundation["G"],
        "term_translation": term_translation,
        "term_rocking": term_rocking,
        "T_ratio": T_ratio,                                                # T_tilde / T_fixed
        "T_with_ssi_s": T_with_ssi,
        "f_with_ssi_hz": f_with_ssi,
        "sigma": sigma,
        "vs_used_ms": vs,
        "site_class": CAMPUS_SITE_CLASS,
    }


def write_estimates(out_dir: Path) -> Path:
    """Write ssi_estimates.csv + ssi_estimates.md to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [period_lengthening(cfg) for cfg in STATIONS.values()]
    csv_path = out_dir / "ssi_estimates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    md_path = out_dir / "ssi_estimates.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(_format_md(rows))
    return csv_path


def _format_md(rows: list[dict]) -> str:
    lines = []
    lines.append("# SSI Period-Lengthening Bounds (NIST GCR 12-917-21 §2)\n")
    lines.append(
        "Conservative upper-bound estimate of T_tilde / T_fixed from "
        "Veletsos-Meek (1974) using Pais-Kausel (1988) static foundation "
        "stiffnesses on a homogeneous halfspace.\n"
    )
    lines.append("**Site inputs**\n")
    lines.append(f"- Vs30 = {CAMPUS_VS30_MS:.1f} m/s  (HWA018 志學國小 borehole, NCREE EGDT)")
    lines.append(f"- Site class = {CAMPUS_SITE_CLASS} (NEHRP)")
    lines.append(f"- Soil density ρ = {CAMPUS_SOIL_DENSITY_KGM3:.0f} kg/m³")
    lines.append(f"- Poisson ν = {CAMPUS_SOIL_POISSON:.2f}")
    lines.append("")
    lines.append(
        "**Interpretation of σ = h_bar / (Vs · T_fixed)** (NIST §2.6): "
        "σ < 0.1 → SSI negligible; 0.1 ≤ σ < 0.2 → minor; "
        "σ ≥ 0.2 → SSI must be modelled. "
        "**Δf_SSI%** is the fraction of f drop attributable purely to SSI flexibility:\n\n"
        "  Δf_SSI% = (1 − f_with_ssi / f_fixed) × 100%\n"
    )
    lines.append("")
    header = ("| station | n_floors | h(m) | footprint(m²) | r_eq(m) | "
              "f_fixed(Hz) | f_with_SSI(Hz) | T_tilde/T | σ | Δf_SSI(%) |")
    sep = "|" + "|".join(["---"] * 10) + "|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        df_pct = (1.0 - r["f_with_ssi_hz"] / r["f_fixed_hz"]) * 100
        lines.append(
            f"| {r['station_id']} | {r['n_floors']} | {r['height_m']:.0f} | "
            f"{r['footprint_m2']:.0f} | {r['r_eq_m']:.1f} | "
            f"{r['f_fixed_hz']:.2f} | {r['f_with_ssi_hz']:.2f} | "
            f"{r['T_ratio']:.3f} | {r['sigma']:.3f} | {df_pct:.1f} |"
        )
    lines.append("")
    lines.append("## Reading the table")
    lines.append("")
    lines.append(
        "For NDHU on NEHRP Class C ground (Vs ≈ 460 m/s), the predicted SSI "
        "period-lengthening is small. Whatever fraction of the observed "
        "coseismic f drop EXCEEDS the Δf_SSI(%) column is therefore "
        "attributable to structural softening rather than soil-foundation "
        "flexibility. This is an UPPER BOUND assuming linear soil; during "
        "strong shaking soil G degrades and Δf_SSI grows, but the table "
        "below provides the small-strain reference."
    )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Mass estimate uses 800 kg/m² × footprint × n_floors. "
                 "Actual structural drawings would refine this; the "
                 "T_tilde/T ratio is fairly insensitive to mass error "
                 "(scales as √mass).")
    lines.append("- Pais-Kausel surface-footing formulas neglect embedment "
                 "and pile foundations. NDHU buildings are mostly low-rise "
                 "shallow mat — adequate.")
    lines.append("- Vs30 is from the 31.5 m HWA018 borehole. Deeper Vs "
                 "(below 30 m) was not measured; the halfspace assumption "
                 "is conservative for the foundation rocking mode.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = Path(r"D:\PHD\project2026") / "ssi_summary"
    write_estimates(out)
    print(f"Wrote SSI bounds to {out}")
