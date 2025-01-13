import colour
import matplotlib.pyplot as plt
import numpy as np
from colour.colorimetry.spectrum import SpectralDistribution
from colour.quality.cri import ColourRendering_Specification_CRI
from scipy.interpolate import PchipInterpolator


def read_spectral_data(file_path: str) -> dict[float, float]:
    spectral_data = {}
    with open(file_path) as file:
        for line in file:
            if line.strip() == "" or line.strip().startswith("nm") or line.startswith("---"):
                continue

            parts = line.split()
            if len(parts) == 2:
                wavelength = float(parts[0])
                intensity = float(parts[1])
                spectral_data[wavelength] = intensity
    return spectral_data


def interpolate_spectral_data(spectral_data: dict[float, float]) -> dict[np.int64, np.float64]:
    wavelengths = np.array(list(spectral_data.keys()))
    intensities = np.array(list(spectral_data.values()))

    need_extrapolation = False
    if wavelengths[-1] < 780:
        print("Warning: Input data does not extend to 780 nm. Extrapolating values.")
        need_extrapolation = True
    if wavelengths[0] > 360:
        print("Warning: Input data does not extend to 360 nm. Extrapolating values.")
        need_extrapolation = True

    interpolator = PchipInterpolator(wavelengths, intensities, extrapolate=need_extrapolation)

    interpolated_wavelengths = np.arange(360, 781, 1)
    interpolated_intensities = interpolator(interpolated_wavelengths)

    return dict(zip(interpolated_wavelengths, interpolated_intensities, strict=True))


def process_spectral_data(file_path: str) -> SpectralDistribution:
    spectral_data = read_spectral_data(file_path)
    interpolated_data = interpolate_spectral_data(spectral_data)
    return colour.SpectralDistribution(interpolated_data, name="Measured Spectrum")


def calculate_cri_and_cct(spectral_distribution: SpectralDistribution) -> tuple[ColourRendering_Specification_CRI, np.float64]:
    cri_results = colour.quality.colour_rendering_index(spectral_distribution, additional_data=True)

    xyz = colour.sd_to_XYZ(spectral_distribution)
    cct = colour.temperature.XYZ_to_CCT_Ohno2013(xyz)[0]

    return cri_results, cct


def calculate_tm30(spectral_distribution: SpectralDistribution) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    tm30_specification = colour.colour_fidelity_index(spectral_distribution, method="ANSI/IES TM-30-18", additional_data=True)
    tm30_rf = tm30_specification.R_f
    tm30_rg = tm30_specification.R_g
    tm30_cct = tm30_specification.CCT
    tm30_duv = tm30_specification.D_uv

    return tm30_rf, tm30_rg, tm30_cct, tm30_duv


def calculate_xy_color(spectral_distribution: SpectralDistribution) -> np.ndarray:
    xyz = colour.sd_to_XYZ(spectral_distribution)

    return colour.XYZ_to_xy(xyz)


def calculate_gamut_polygon(spectral_distribution: SpectralDistribution) -> np.ndarray:
    spectral_wavelengths = spectral_distribution.wavelengths
    spectral_values = spectral_distribution.values

    spectral_xy = []
    max_intensity = max(spectral_values) / 2  # Use half the maximum intensity as the threshold

    for wl, intensity in zip(spectral_wavelengths, spectral_values, strict=True):
        if intensity >= max_intensity:  # Process only wavelengths with intensity >= half max
            domain = np.arange(360, 781, 1)
            values = np.zeros(domain.shape)
            if wl in domain:
                values[np.where(domain == wl)[0][0]] = intensity

            mono_sd = colour.SpectralDistribution(dict(zip(domain, values, strict=True)), name=f"{wl} nm")

            xyz = colour.sd_to_XYZ(mono_sd)
            xy = colour.XYZ_to_xy(xyz)
            spectral_xy.append(xy)

    return np.array(spectral_xy)


def plot_color_gamut_with_polygon(spectral_distribution: SpectralDistribution) -> None:
    # Get the xy chromaticity of the light source
    xy = calculate_xy_color(spectral_distribution)

    # Calculate the gamut polygon
    gamut_polygon = calculate_gamut_polygon(spectral_distribution)

    # Plot the chromaticity diagram
    colour.plotting.plot_planckian_locus_in_chromaticity_diagram_CIE1931(show=False)

    # Plot the gamut polygon with clear color and a black border
    plt.fill(
        gamut_polygon[:, 0],
        gamut_polygon[:, 1],
        color="none",
        edgecolor="black",
        alpha=0.8,
        label="Light Source Gamut",
    )

    # Plot the light source's chromaticity point
    plt.scatter(xy[0], xy[1], color="red", label="Light Source", zorder=10)

    # Add labels and legend
    plt.legend()
    plt.title("CIE 1931 Chromaticity Diagram with Light Source Gamut")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    plt.show()


def rate_duv(duv: np.float64) -> str:
    if abs(duv) <= 0.001:
        deviation = "Virtually imperceptible"
        direction = "aligned with the Planckian locus"
        application = "suitable for most applications requiring white light."
    elif abs(duv) <= 0.005:
        deviation = "Slightly off"
        direction = "towards green" if duv > 0 else "towards magenta"
        application = "acceptable in high-quality lighting applications."
    elif abs(duv) <= 0.015:
        deviation = "Noticeably off"
        direction = "towards green" if duv > 0 else "towards magenta"
        application = "usable but may not meet standards for critical lighting."
    elif abs(duv) <= 0.03:
        deviation = "Significantly off"
        direction = "towards green" if duv > 0 else "towards magenta"
        application = "unsuitable for applications requiring precise color rendering."
    else:
        deviation = "Strongly off"
        direction = "towards green" if duv > 0 else "towards magenta"
        application = "highly undesirable for most applications."

    return f"{deviation} {direction}; {application}"


def calculate_din6169_re(cri_results: ColourRendering_Specification_CRI) -> float:
    re_values = [value.Q_a for value in cri_results.Q_as.values()]

    return sum(re_values) / len(re_values) if re_values else 0.0


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
        return

    file_path = sys.argv[1]

    spectral_distribution = process_spectral_data(file_path)

    cri_results, cct = calculate_cri_and_cct(spectral_distribution)

    din6169_re = calculate_din6169_re(cri_results)

    tm30_rf, tm30_rg, tm30_cct, tm30_duv = calculate_tm30(spectral_distribution)

    xy = calculate_xy_color(spectral_distribution)

    print(f"General CRI (Ra): {cri_results.Q_a:.2f}")
    print(f"DIN 6169 Re: {din6169_re:.2f}")

    print("Extended CRI (R values):")
    for _i, (key, value) in enumerate(cri_results.Q_as.items()):
        print(f"  R{key}: {value.Q_a:.2f}")

    print(f"Correlated Color Temperature (CCT, Ohno 2013): {cct:.2f} K")

    # Print TM-30 Results
    print(f"TM-30 Fidelity Index (Rf): {tm30_rf:.2f}")
    print(f"TM-30 Gamut Index (Rg): {tm30_rg:.2f}")
    print(f"TM-30 CCT: {tm30_cct:.2f} K")
    print(f"TM-30 Duv: {tm30_duv:.6f}")
    print(f"Duv Rating: {rate_duv(tm30_duv)}")

    print(f"CIE 1931 Chromaticity Coordinates: x={xy[0]:.4f}, y={xy[1]:.4f}")

    plot_color_gamut_with_polygon(spectral_distribution)


if __name__ == "__main__":
    main()
