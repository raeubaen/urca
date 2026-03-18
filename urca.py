import numpy as np
import tensorflow as tf
from scipy.interpolate import RegularGridInterpolator
import ROOT
import math
import matplotlib.pyplot as plt

def bernstein_basis(m, degree):
    m_min = tf.reduce_min(m)
    m_max = tf.reduce_max(m)
    
    x = (m - m_min) / (m_max - m_min + 1e-6)  # normalize to [0,1]
    x = tf.reshape(x, (-1, 1))  # [n_m,1]

    basis = []
    for k in range(degree + 1):
        coeff = math.comb(degree, k)
        term = coeff * (x**k) * ((1 - x)**(degree - k))
        basis.append(term)

    return tf.concat(basis, axis=1)  # [n_m, degree+1]

def cut_th3_region(th3, f1_min=-np.inf, f1_max=np.inf, f2_min=-np.inf, f2_max=np.inf):
    """
    Returns a new TH3 with only the bins where
    f1_min <= f1 <= f1_max and f2_min <= f2 <= f2_max.
    Bins outside are removed in the arrays.
    """
    # Extract original grids
    x_centers = np.array([th3.GetXaxis().GetBinCenter(i+1) for i in range(th3.GetNbinsX())])
    y_centers = np.array([th3.GetYaxis().GetBinCenter(j+1) for j in range(th3.GetNbinsY())])
    z_centers = np.array([th3.GetZaxis().GetBinCenter(k+1) for k in range(th3.GetNbinsZ())])

    # Select bins inside the cut
    y_mask = (y_centers >= f1_min) & (y_centers <= f1_max)
    z_mask = (z_centers >= f2_min) & (z_centers <= f2_max)

    y_selected = y_centers[y_mask]
    z_selected = z_centers[z_mask]

    values = np.zeros((len(x_centers), len(y_selected), len(z_selected)))

    for i, xi in enumerate(x_centers):
        for j_idx, j in enumerate(np.where(y_mask)[0]):
            for k_idx, k in enumerate(np.where(z_mask)[0]):
                # Get the global bin number in TH3
                global_bin = th3.GetBin(int(i+1), int(j+1), int(k+1))
                values[i, j_idx, k_idx] = th3.GetBinContent(global_bin)

    return x_centers, y_selected, z_selected, values


def fit_background_bernstein(m, B_m_c, sb_mask_f, degree=3, epsilon=1e-6):
    """
    Fit a Bernstein polynomial to the background in sidebands, using per-category Poisson errors
    m: [n_m] mass bins
    B_m_c: [n_m, N_cat] counts per category
    sb_mask_f: [n_m] sideband mask (0/1)
    """
    X = bernstein_basis(m, degree)  # [n_m, K]
    n_m, N_cat = B_m_c.shape
    K = X.shape[1]

    B_fit = []

    for c in range(N_cat):
        y_c = B_m_c[:, c:c+1]  # [n_m, 1]

        # weights: 1 / variance
        w_c = sb_mask_f / (y_c[:, 0] + epsilon)  # [n_m]

        # Apply weights
        Xw = X * tf.expand_dims(w_c, axis=1)  # [n_m, K]
        Yw = y_c * tf.expand_dims(w_c, axis=1)  # [n_m, 1]

        # Normal equations
        XT = tf.transpose(X)  # [K, n_m]
        XT_W_X = tf.matmul(XT, Xw) + 1e-6 * tf.eye(K)  # regularization
        XT_W_Y = tf.matmul(XT, Yw)

        # Solve
        coeffs_c = tf.linalg.solve(XT_W_X, XT_W_Y)  # [K,1]

        # Evaluate fit
        B_fit_c = tf.matmul(X, coeffs_c)  # [n_m,1]
        B_fit.append(B_fit_c)

    # Stack all categories: [n_m, N_cat]
    B_fit = tf.concat(B_fit, axis=1)
    return B_fit


def main(th3_signal, th3_bkg):
    # Example cut: f1 in [0,150], f2 in [0.6,1.0]
    m_centers, f1_centers, f2_centers, rho_signal_values = cut_th3_region(th3_signal,
                                                               f1_min=0, f1_max=150,
                                                               f2_min=0.6, f2_max=1.0)

    _, _, _, rho_bkg_values = cut_th3_region(th3_background,
                                          f1_min=0, f1_max=150,
                                          f2_min=0.6, f2_max=1.0)

    # Interpolators
    rho_signal_interp = RegularGridInterpolator((m_centers, f1_centers, f2_centers), rho_signal_values,
                                                method='linear', bounds_error=False, fill_value=0)
    rho_bkg_interp = RegularGridInterpolator((m_centers, f1_centers, f2_centers), rho_bkg_values,
                                            method='linear', bounds_error=False, fill_value=0)

    # -----------------------------
    # Step 2: Define softmax neural network
    # -----------------------------
    N_categories = 2  # example
    hidden_units = 128

    nn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),  # only f1 as input
        tf.keras.layers.Dense(hidden_units, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dense(hidden_units, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dense(1)  # curve output f2 = g(f1)
    ])


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    # -----------------------------
    # Step 3: Prepare evaluation grid in f1,f2
    # -----------------------------
    f1_grid, f2_grid = np.meshgrid(f1_centers, f2_centers, indexing='ij')
    f1_flat = f1_grid.flatten()
    f2_flat = f2_grid.flatten()
    coords_flat = np.stack([f1_flat, f2_flat], axis=-1)

    # Convert to tf.Tensor
    coords_tf = tf.convert_to_tensor(coords_flat, dtype=tf.float32)

    m_tf = tf.convert_to_tensor(m_centers, dtype=tf.float32)

    f1_min, f1_max = f1_centers[0], f1_centers[-1]
    f2_min, f2_max = f2_centers[0], f2_centers[-1]

    # Compute middle of normalized f2
    b0 = 0.5  # normalized midpoint in [0,1]
    m0 = 0.0  # initial slope (flat line)

    # Last layer of the network
    last_layer = nn_model.layers[-1]

    # Small random weights so curve starts almost flat
    kernel_init = tf.random.normal(last_layer.kernel.shape, mean=0.0, stddev=0.01)
    last_layer.kernel.assign(kernel_init)

    # Bias set to normalized middle
    last_layer.bias.assign(tf.constant([b0], dtype=tf.float32))



    m_low, m_high = 120.0, 130.0  # adjust if needed

    sr_mask = tf.logical_and(m_tf >= m_low, m_tf <= m_high)
    sb_mask = tf.logical_not(sr_mask)

    sr_mask_f = tf.cast(sr_mask, tf.float32)
    sb_mask_f = tf.cast(sb_mask, tf.float32)

    # -----------------------------
    # Step 4: Training loop
    # -----------------------------
    n_epochs = 50

    loss_history = []
    metric_history = []

    # 4b. Compute weighted 1D mass projections
    # Evaluate rho on the f1,f2 grid for all m
    rho_signal_grid = np.zeros((len(m_centers), len(f1_centers), len(f2_centers)))
    rho_bkg_grid = np.zeros_like(rho_signal_grid)
    for i, m in enumerate(m_centers):
        pts = np.stack([np.full(f1_flat.shape, m), f1_flat, f2_flat], axis=-1)
        rho_signal_grid[i] = rho_signal_interp(pts).reshape(len(f1_centers), len(f2_centers))
        rho_bkg_grid[i] = rho_bkg_interp(pts).reshape(len(f1_centers), len(f2_centers))

    # Convert to tf.Tensor
    rho_signal_tf = tf.convert_to_tensor(rho_signal_grid, dtype=tf.float32)
    rho_bkg_tf = tf.convert_to_tensor(rho_bkg_grid, dtype=tf.float32)


    # f1_tf and f2_tf: [n_points,1]
    # coords_tf[:,0] = f1, coords_tf[:,1] = f2
    f1_tf = coords_tf[:,0:1]
    f2_tf = coords_tf[:,1:2]

    # Normalize f2 to [0,1]
    f2_min, f2_max = f2_centers[0], f2_centers[-1]
    f2_scaled = (f2_tf - f2_min) / (f2_max - f2_min)

    # Optional: normalize f1 too
    f1_min, f1_max = f1_centers[0], f1_centers[-1]
    f1_scaled = (f1_tf - f1_min) / (f1_max - f1_min)

    for epoch in range(n_epochs):
        with tf.GradientTape() as tape:

            # Curve network output in [0,1] using sigmoid
            g_f1 = tf.sigmoid(nn_model(f1_scaled))  # [n_points,1]

            # Soft assignment along curve
            tau = 0.05  # smaller -> sharper assignment
            p1 = tf.sigmoid((f2_scaled - g_f1)/tau)  # probability for category 1
            p0 = 1.0 - p1
            p_grid = tf.concat([p0, p1], axis=-1)

            # Stack probabilities into [n_points, 2]
            p_grid = tf.concat([p0, p1], axis=-1)

            # Reshape for tensordot over f1/f2 grid
            p_grid_2d = tf.reshape(p_grid, (len(f1_centers), len(f2_centers), 2))

            # Use p_grid_2d in S/B projections as before
            S_m_c = tf.tensordot(rho_signal_tf, p_grid_2d, axes=[[1,2],[0,1]])
            B_m_c = tf.tensordot(rho_bkg_tf, p_grid_2d, axes=[[1,2],[0,1]])


            # 4c. Compute chi2 metric
            epsilon = 1e-6

            # pseudo-data signal-only - so unbiased by possibly bad bkg fits
            D_m_c = S_m_c

            # fit background from sidebands only
            B_fit = fit_background_bernstein(m_tf, B_m_c, sb_mask_f)

            # restrict to signal region
            D_sr = D_m_c * tf.expand_dims(sr_mask_f, axis=-1)
            B_sr = B_fit * tf.expand_dims(sr_mask_f, axis=-1)

            # chi2 in SR
            chi2_c = tf.reduce_sum((D_sr)**2 / (B_sr + epsilon), axis=0)

            metric = tf.reduce_sum(chi2_c)

            # 4e. Total loss
            loss = -metric

            min_frac = 0.05
            frac_per_cat = tf.reduce_sum(p_grid) / tf.size(p_grid, out_type=tf.float32)
            penalty = tf.reduce_sum(tf.nn.relu(min_frac - frac_per_cat))
            loss += 0.1 * penalty

            # Compute approximate derivative along f1
            g_f1 = tf.sigmoid(nn_model(f1_scaled))  # [n_points,1]

            dg_df1 = g_f1[1:] - g_f1[:-1]           # finite difference
            slope_penalty = tf.reduce_mean(tf.square(dg_df1))  # mean squared slope

            # Add to loss
            loss += 0.01 * slope_penalty  # scale factor to tune

            loss_history.append(loss.numpy())
            metric_history.append(metric.numpy())

            # --- Compute signal/background counts in the SR per category ---
            sr_mask_f = tf.cast(tf.logical_and(m_tf >= m_low, m_tf <= m_high), tf.float32)

            S_counts = tf.reduce_sum(S_m_c * tf.expand_dims(sr_mask_f, axis=-1), axis=0)
            B_counts = tf.reduce_sum(B_m_c * tf.expand_dims(sr_mask_f, axis=-1), axis=0)

            S_counts_np = S_counts.numpy()
            B_counts_np = B_counts.numpy()
            for c in range(N_categories):
                print(f"Epoch {epoch}: Category {c}: S={S_counts_np[c]:.2f}, B={B_counts_np[c]:.2f}")


        # 4f. Apply gradients
        grads = tape.gradient(loss, nn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, nn_model.trainable_variables))

        print(f"Epoch {epoch}: loss = {loss.numpy():.4f}, metric = {metric.numpy():.4f}")


    # -----------------------------
    # After training: HARD category evaluation
    # -----------------------------

    plt.figure() 
    plt.plot(loss_history, label="Loss") 
    plt.plot(metric_history, label="Metric") 
    plt.xlabel("Epoch") 
    plt.ylabel("Value") 
    plt.legend() 
    plt.title("Training evolution") 
    plt.savefig("training.png")


    # Compute the network output (curve) on the evaluation grid
    g_f1 = tf.sigmoid(nn_model(f1_scaled))  # [n_points, 1]

    # Evaluate curve on the f1 grid
    f1_grid_plot = np.linspace(0,1,len(f1_centers)).reshape(-1,1)
    g_f1_plot = tf.sigmoid(nn_model(tf.convert_to_tensor(f1_grid_plot, dtype=tf.float32))).numpy().flatten()

    # Rescale to original f2 range
    g_f1_rescaled = g_f1_plot * (f2_max - f2_min) + f2_min

    plt.figure(figsize=(6,5))
    plt.plot(f1_centers, g_f1_rescaled, color='red', linewidth=2, label='Learned curve')
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("Learned f2 = g(f1) after training")
    plt.grid(True)
    plt.legend()
    plt.savefig("learned_curve.png")
    plt.close()

    # HARD assignment along f2
    hard_grid = (f2_scaled > g_f1).numpy().astype(int).reshape(len(f1_centers), len(f2_centers))
    # If N_categories > 2, generalize with multiple thresholds/curves

    # Signal region window
    m_low_eval, m_high_eval = 123.0, 127.0
    sr_mask_eval = (m_centers >= m_low_eval) & (m_centers <= m_high_eval)

    # Preallocate S and B arrays per category
    S_hard = np.zeros(N_categories)
    B_hard = np.zeros(N_categories)

    # Mass spectra for plotting
    S_mass = np.zeros((len(m_centers), N_categories))
    B_mass = np.zeros((len(m_centers), N_categories))

    for c in range(N_categories):
        # Boolean mask for this category in f1,f2
        cat_mask = (hard_grid == c).astype(np.float32)  # (n_f1, n_f2)

        # Project rho_signal and rho_bkg to 1D mass spectra for this category
        S_mass[:, c] = tf.tensordot(rho_signal_tf, cat_mask, axes=[[1,2],[0,1]]).numpy()
        B_mass[:, c] = tf.tensordot(rho_bkg_tf, cat_mask, axes=[[1,2],[0,1]]).numpy()

        # Integrate in evaluation window
        S_hard[c] = np.sum(S_mass[sr_mask_eval, c])
        B_hard[c] = np.sum(B_mass[sr_mask_eval, c])

    # Print counts per category
    print("Category-wise counts in [123,127] GeV (HARD cuts):")
    for c in range(N_categories):
        print(f"Category {c}: S = {S_hard[c]:.2f}, B = {B_hard[c]:.2f}")

    # Ensure B_fit_np is available and in NumPy
    B_fit_np = B_fit.numpy()  # shape: (n_m, N_cat)

    # Ensure B_fit_np is available
    B_fit_np = B_fit.numpy()  # shape (n_m, N_cat)

    m_low_sb, m_high_sb = 115, 135

    for c in range(N_categories):
        plt.figure(figsize=(6,5))

        # Masks
        inside_sr_mask = (m_centers >= m_low_sb) & (m_centers <= m_high_sb)
        outside_sr_mask = ~inside_sr_mask


        # Split sidebands into two continuous segments
        left_sb_mask = m_centers < m_low_sb
        right_sb_mask = m_centers > m_high_sb

        # 1️⃣ Signal + B_fit in SR
        plt.step(m_centers[inside_sr_mask],
                 S_mass[inside_sr_mask, c] + B_fit_np[inside_sr_mask, c],
                 where='mid', label='S + B_fit (SR)', color='C0')


        # 2️⃣ Real B only in sidebands (split)
        if np.any(left_sb_mask):
            plt.step(m_centers[left_sb_mask],
                     B_mass[left_sb_mask, c],
                     where='mid', color='C1', linestyle='--')
        if np.any(right_sb_mask):
            plt.step(m_centers[right_sb_mask],
                     B_mass[right_sb_mask, c],
                     where='mid', color='C1', linestyle='--', label='B (sidebands)')

        # 3️⃣ B_fit in the full range
        plt.step(m_centers[:],
                 B_fit_np[:, c],
                 where='mid', label='B_fit (full range)', color='C2')



        # Highlight signal region
        plt.axvspan(m_low_sb, m_high_sb, alpha=0.2, color='gray', label='Signal Region')

        plt.xlabel("Mass [GeV]")
        plt.ylabel("Events")
        plt.title(f"Category {c} (HARD assignment)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"mass_spectra_hard_cat{c}.png")
        plt.close()


if __name__ == "__main__":
    # Paths
    bkg_path = "../CMSSW_14_1_0_pre4/src/flashggFinalFit/ReCat/data_th3_recat.root"


    sig_path = "../CMSSW_14_1_0_pre4/src/flashggFinalFit/ReCat/sig_th3_recat.root"

    # Open files
    f_bkg = ROOT.TFile.Open(bkg_path)
    f_sig = ROOT.TFile.Open(sig_path)

    if not f_bkg or f_bkg.IsZombie():
        raise RuntimeError(f"Cannot open background file: {bkg_path}")
    if not f_sig or f_sig.IsZombie():
        raise RuntimeError(f"Cannot open signal file: {sig_path}")

    # Retrieve TH3
    th3_background = f_bkg.Get("h_0")
    th3_signal = f_sig.Get("h_0")

    if not th3_background:
        raise RuntimeError("TH3 'h_0' not found in background file")
    if not th3_signal:
        raise RuntimeError("TH3 'h_0' not found in signal file")

    # Detach from file (important to avoid ROOT ownership issues)
    th3_background = th3_background.Clone("th3_background")
    th3_signal = th3_signal.Clone("th3_signal")

    th3_background.SetName("bkg_th3")
    th3_signal.SetName("sig_th3")

    th3_background.SetDirectory(0)
    th3_signal.SetDirectory(0)

    # -----------------------------
    # Rescale signal luminosity
    # -----------------------------
    lumi_data = 67.0
    lumi_signal = 1.0

    scale_factor = lumi_data / lumi_signal  # = 67
    th3_signal.Scale(scale_factor)

    print(f"[INFO] Signal scaled by factor {scale_factor}")

    # Pass filtered TH3 to main
    main(th3_signal, th3_background)
