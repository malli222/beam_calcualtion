import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title=" Beam Divergence in Vacuum & Atmosphere", layout="wide")

tab_main, tab_converter, tab_lens = st.tabs([
    " Beam Divergence Tool",
    " Loss % ⇄ dB Converter",
    " Interactive Loss Estimator"
])

# ------------------------- TAB 1: BEAM TOOL -------------------------
with tab_main:
    st.title(" Beam Divergence & Diffraction Loss Tool")

    # ------------------------- INPUTS -------------------------
    st.sidebar.header(" Input Parameters")
    λ_nm = st.sidebar.number_input("Wavelength (nm)", min_value=400, max_value=2000, value=1550)
    λ = λ_nm * 1e-9  # Convert to meters

    beam_diameter_mm = st.sidebar.slider("Beam Waist Diameter (mm)", min_value=1, max_value=300, value=20)
    w0 = (beam_diameter_mm / 1000) / 2  # Beam waist radius (m)

    z = st.sidebar.slider("Propagation Distance z (m)", min_value=1, max_value=10000, value=5000)
    rx_aperture_mm = st.sidebar.slider("RX Lens Aperture (mm)", min_value=10, max_value=300, value=100)
    rx_aperture_m = rx_aperture_mm / 1000

    Cn2 = st.sidebar.number_input("Cn² (Refractive Index Structure Constant)", min_value=1e-17, max_value=1e-12, value=1e-14, format="%.1e")


    # ------------------------- FORMULAS AND THEORY -------------------------
    st.header(" Beam Propagation Theory")

    st.subheader(" Beam Divergence Angle")
    theta = λ / (np.pi * w0)
    theta_mrad = theta * 1e3
    st.latex(r"\theta = \frac{\lambda}{\pi w_0}")
    st.markdown(f"**Divergence Angle:** `{theta:.2e} rad` or `{theta_mrad:.2f} mrad`")

    st.subheader(" Scintillation Index (Weak Turbulence)")
    k = 2 * np.pi / λ
    sigma_I2 = 1.23 * Cn2 * k**(7/6) * z**(11/6)
    st.latex(r"\sigma_I^2 = 1.23 \, C_n^2 \, k^{7/6} \, z^{11/6}")
    st.markdown(f"**Estimated Scintillation Index σ²:** `{sigma_I2:.2f}`")
    st.markdown("""
    **Interpretation**:
    - If σ² < 1 → Weak turbulence (good conditions)  
    - If σ² > 1 → Moderate to strong turbulence (AO needed)
    """)


    # ------------------------- SHARED CALCULATIONS -------------------------
    z_R = (np.pi * w0**2) / λ  # Rayleigh length (m)

    # ------------------------- SECTION 1: VACUUM -------------------------
    st.header(" Section 1: Beam Propagation in Vacuum (No Turbulence)")

    w_vac = w0 * np.sqrt(1 + (z / z_R)**2)
    beam_diameter_vac_mm = 2 * w_vac * 1000

    clip_vac = rx_aperture_m / w_vac
    η_vac = 1 - np.exp(-2 * clip_vac**2)
    loss_vac_db = -10 * np.log10(np.clip(η_vac, 1e-10, 1))

    st.markdown(f"**Rayleigh Length:** `{z_R:.2f} m`")
    st.markdown(f"**Beam Diameter at {z} m (Vacuum):** `{beam_diameter_vac_mm:.2f} mm`")
    st.markdown(f"**Transmission Efficiency:** `{η_vac*100:.2f}%`")
    st.markdown(f"**Diffraction Loss:** `{loss_vac_db:.2f} dB`")

    # ------------------------- SECTION 2: ATMOSPHERIC TURBULENCE -------------------------
    st.header(" Section 2: Beam Propagation with Atmospheric Turbulence")

    w_total = np.sqrt(
        w0**2 + (λ * z / (np.pi * w0))**2 + 4 * np.pi**2 * Cn2 * z**3 * w0**(-1/3)
    )
    beam_diameter_turb_mm = 2 * w_total * 1000

    clip_turb = rx_aperture_m / w_total
    η_turb = 1 - np.exp(-2 * clip_turb**2)
    loss_turb_db = -10 * np.log10(np.clip(η_turb, 1e-10, 1))

    st.markdown(f"**Beam Diameter at {z} m (With Turbulence):** `{beam_diameter_turb_mm:.2f} mm`")
    st.markdown(f"**Transmission Efficiency (With Turbulence):** `{η_turb*100:.2f}%`")
    st.markdown(f"**Diffraction Loss (With Turbulence):** `{loss_turb_db:.2f} dB`")

    # ------------------------- PLOT COMPARISON -------------------------
    st.header(" Beam Profile Comparison")

    x_vals = np.linspace(1, 10000, 1000)
    w_vac_all = w0 * np.sqrt(1 + (x_vals / z_R)**2)
    w_turb_all = np.sqrt(w0**2 + (λ * x_vals / (np.pi * w0))**2 + 4 * np.pi**2 * Cn2 * x_vals**3 * w0**(-1/3))

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(x_vals, 2 * w_vac_all * 1000, label="Vacuum (No Turbulence)", linestyle='--')
    ax1.plot(x_vals, 2 * w_turb_all * 1000, label="With Turbulence", color='blue')
    ax1.axhline(rx_aperture_mm, color='red', linestyle='--', label="RX Aperture")
    ax1.axvline(z, color='orange', linestyle=':', label=f"RX at {z} m")
    ax1.set_xlabel("Distance [m]")
    ax1.set_ylabel("Beam Diameter [mm]")
    ax1.set_title("Beam Diameter vs Distance")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    # ------------------------- LOSS COMPARISON -------------------------
    st.header(" Diffraction Loss Comparison")

    η_vac_all = 1 - np.exp(-2 * (rx_aperture_m / w_vac_all)**2)
    loss_vac_all_db = -10 * np.log10(np.clip(η_vac_all, 1e-10, 1))

    η_turb_all = 1 - np.exp(-2 * (rx_aperture_m / w_turb_all)**2)
    loss_turb_all_db = -10 * np.log10(np.clip(η_turb_all, 1e-10, 1))

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(x_vals, loss_vac_all_db, linestyle='--', label="Vacuum")
    ax2.plot(x_vals, loss_turb_all_db, label="With Turbulence", color='purple')
    ax2.set_xlabel("Distance [m]")
    ax2.set_ylabel("Diffraction Loss [dB]")
    ax2.set_title("Loss vs Distance")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)


    # ------------------------- FIXED DISTANCE COMPARISON -------------------------
    # ------------------------- FIXED DISTANCE COMPARISON WITH BOUNDS -------------------------
    st.header(" RX Beam Size Bounds at 5 km for TX Beam Diameter 25–700 mm")

    fixed_z = 5000  # meters
    tx_diams_mm = np.linspace(25, 700, 500)
    w0_vals = (tx_diams_mm / 1000) / 2  # meters
    z_R_vals = (np.pi * w0_vals**2) / λ

    # Lower Bound (Vacuum)
    w_rx_vac = w0_vals * np.sqrt(1 + (fixed_z / z_R_vals)**2)

    # Turbulence for different Cn2
    Cn2_vals = [1e-16, 1e-15, 1e-14]
    colors = ['green', 'blue', 'red']
    labels = [r"$C_n^2 = 10^{-16}$", r"$C_n^2 = 10^{-15}$", r"$C_n^2 = 10^{-14}$"]

    w_rx_turb = []
    for Cn2_val in Cn2_vals:
        w_t = np.sqrt(w0_vals**2 + (λ * fixed_z / (np.pi * w0_vals))**2 +
                    4 * np.pi**2 * Cn2_val * fixed_z**3 * w0_vals**(-1/3))
        w_rx_turb.append(w_t)

    # Upper bound: use Cn2 = 10^-14
    Cn2_max = 1e-14
    w_upper = np.sqrt(
        w0_vals**2 + (λ * fixed_z / (np.pi * w0_vals))**2 + 4 * np.pi**2 * Cn2_max * fixed_z**3 * w0_vals**(-1/3)
    )
    w_lower = w_rx_vac

    beam_d_lower_mm = 2 * w_lower * 1000
    beam_d_upper_mm = 2 * w_upper * 1000

    # Plot
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    ax5.fill_between(tx_diams_mm, beam_d_lower_mm, beam_d_upper_mm, color='gray', alpha=0.3, label="Beam Size Bounds")

    # Main curves
    ax5.plot(tx_diams_mm, 2 * w_rx_vac * 1000, label="Vacuum", linestyle="--", color="black")
    for i in range(len(Cn2_vals)):
        ax5.plot(tx_diams_mm, 2 * w_rx_turb[i] * 1000, label=labels[i], color=colors[i])

    ax5.set_title("RX Beam Diameter vs TX Beam Diameter at 5 km (with Bounds)")
    ax5.set_xlabel("TX Beam Diameter [mm]")
    ax5.set_ylabel("RX Beam Diameter [mm]")
    ax5.grid(True)
    ax5.legend()
    st.pyplot(fig5)


    # ------------------------- RECOMMENDED TX BEAM DIAMETER SECTION -------------------------
    st.header(" Recommended TX Beam Diameter for 5 km Link (±10% RX Aperture Tolerance)")

    # Use same λ and Cn² from earlier sidebar inputs
    fixed_z = 5000  # meters
    D_rx_nominal_mm = st.number_input("RX Lens Aperture (mm)", min_value=50, max_value=300, value=100)
    D_rx_nominal = D_rx_nominal_mm / 1000  # meters
    D_min = 0.9 * D_rx_nominal
    D_max = 1.1 * D_rx_nominal

    # Sweep TX beam diameters
    tx_diams_mm_sweep = np.linspace(10, 300, 1000)
    w0_vals_sweep = (tx_diams_mm_sweep / 1000) / 2  # meters

    # Calculate RX beam diameter using full divergence model
    w_rx_all = np.sqrt(
        w0_vals_sweep**2 +
        (λ * fixed_z / (np.pi * w0_vals_sweep))**2 +
        4 * np.pi**2 * Cn2 * fixed_z**3 * w0_vals_sweep**(-1/3)
    )
    beam_d_rx_all = 2 * w_rx_all  # in meters

    # Find valid TX beam range
    within_tol_idx = np.where((beam_d_rx_all >= D_min) & (beam_d_rx_all <= D_max))[0]

    if len(within_tol_idx) > 0:
        tx_min = tx_diams_mm_sweep[within_tol_idx[0]]
        tx_max = tx_diams_mm_sweep[within_tol_idx[-1]]
        st.success(f" Recommended TX Beam Diameter Range: **{tx_min:.1f} mm to {tx_max:.1f} mm**")
    else:
        st.error(" No TX beam diameter satisfies the ±10% RX lens tolerance at 5 km for given conditions.")

    # Optional Plot
    st.subheader(" RX Beam Diameter vs TX Beam Diameter (with Tolerance Band)")

    fig6, ax6 = plt.subplots(figsize=(10, 5))
    ax6.plot(tx_diams_mm_sweep, beam_d_rx_all * 1000, label="RX Beam Diameter [mm]", color='blue')
    ax6.axhline(D_rx_nominal_mm, linestyle='--', color='green', label="RX Aperture")
    ax6.axhline(D_min * 1000, linestyle='--', color='orange', label="-10% Tolerance")
    ax6.axhline(D_max * 1000, linestyle='--', color='red', label="+10% Tolerance")
    ax6.set_xlabel("TX Beam Diameter [mm]")
    ax6.set_ylabel("RX Beam Diameter [mm]")
    ax6.set_title("RX Beam Diameter vs TX Beam Diameter at 5 km")
    ax6.grid(True)
    ax6.legend()
    st.pyplot(fig6)


with tab_converter:
    st.title("📉 Loss % ⇄ dB Converter for Gaussian Beam")

    converter_tab1, converter_tab2, converter_tab3 = st.tabs([
        "Loss % → dB",
        "dB → Loss %",
        "Beam Diameter → Loss (Gaussian)"
    ])

    with converter_tab1:
        st.subheader("Convert Loss % to dB")
        loss_pct = st.number_input("Enter Loss Percentage (%)", min_value=0.0, max_value=100.0, value=10.0, key="loss_pct")
        eta = 1 - (loss_pct / 100)
        db_loss = -10 * np.log10(np.clip(eta, 1e-10, 1))
        st.markdown(f"**Result:** `{db_loss:.2f} dB` attenuation for {loss_pct:.2f}% loss")

    with converter_tab2:
        st.subheader("Convert dB to Loss %")
        db_val = st.number_input("Enter dB Loss", min_value=0.0, max_value=100.0, value=3.0, key="db_val")
        eta_recovered = 10 ** (-db_val / 10)
        loss_pct_recovered = (1 - eta_recovered) * 100
        st.markdown(f"**Result:** `{loss_pct_recovered:.2f}%` loss for {db_val:.2f} dB")

    with converter_tab3:
        st.subheader("Calculate Loss from Beam and RX Aperture Diameter (Gaussian Beam)")

        D_beam_mm = st.number_input("Beam Diameter at RX (mm)", min_value=1.0, max_value=1000.0, value=100.0, key="beam_dia")
        D_rx_mm = st.number_input("Receiver Aperture Diameter (mm)", min_value=1.0, max_value=1000.0, value=50.0, key="rx_dia")

        w_beam = (D_beam_mm / 2) / 1000  # in meters
        r_rx = (D_rx_mm / 2) / 1000      # in meters

        clip_ratio = r_rx / w_beam
        eta = 1 - np.exp(-2 * clip_ratio**2)
        loss_db = -10 * np.log10(np.clip(eta, 1e-10, 1))

        st.markdown(f"""
        **Fraction of Power Captured (η):** `{eta*100:.2f}%`  
        **Diffraction Loss:** `{loss_db:.2f} dB`
        """)
with tab_lens:
    st.title("📊 Beam Loss Estimator: Input/Output Beam Size & Turbulence")

    st.markdown("""
    This tool calculates **Diffraction Loss** using Gaussian beam approximation  
    and estimates **Coupling Loss** based on Adaptive Optics (AO) usage.
    """)

    # Inputs
    input_dia = st.number_input("Input Beam Diameter at TX (mm)", min_value=1.0, max_value=1000.0, value=120.0)
    output_dia = st.number_input("Beam Diameter at RX (mm)", min_value=1.0, max_value=1000.0, value=650.0)
    rx_aperture = st.number_input("RX Aperture Diameter (mm)", min_value=1.0, max_value=1000.0, value=120.0)

    turbulence_level = st.selectbox("Turbulence Level", ["Strong", "Moderate", "Weak"])
    use_ao = st.checkbox("Use Adaptive Optics (AO)?", value=True)

    # Calculations
    w = (output_dia / 2) / 1000  # beam waist in meters
    r_rx = (rx_aperture / 2) / 1000  # receiver radius in meters
    clip_ratio = r_rx / w
    eta = 1 - np.exp(-2 * clip_ratio**2)
    diffraction_loss_db = -10 * np.log10(np.clip(eta, 1e-10, 1))

    # Coupling efficiency (as decimal)
    if use_ao:
        coupling_dict = {"Strong": 0.50, "Moderate": 0.75, "Weak": 0.90}
    else:
        coupling_dict = {"Strong": 0.01, "Moderate": 0.05, "Weak": 0.10}

    coupling_eff = coupling_dict[turbulence_level]
    total_eff = eta * coupling_eff
    total_loss_db = -10 * np.log10(np.clip(total_eff, 1e-10, 1))

    # Results
    st.markdown("### 🔍 Results")
    st.markdown(f"**Diffraction Transmission Efficiency:** `{eta*100:.2f}%`")
    st.markdown(f"**Coupling Efficiency (based on AO & turbulence):** `{coupling_eff*100:.2f}%`")
    st.markdown(f"**Total Effective Transmission Efficiency:** `{total_eff*100:.2f}%`")
    st.markdown(f"**→ Diffraction Loss:** `{diffraction_loss_db:.2f} dB`")
    st.markdown(f"**→ Total Loss (Diffraction + Coupling):** `{total_loss_db:.2f} dB`")


