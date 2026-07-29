"""Rebuild Figure 5 with real magnitude/phase slices from your FastMRI data."""
import os, numpy as np, matplotlib.pyplot as plt, h5py

PROST_PATH = "/Volumes/Research/fastmridatasets/prostate/fastMRI_prostate_DIFF_IDS_001_011/file_prostate_AXDIFF_002.h5"
BREAST_PATH = "/Volumes/Research/fastmridatasets/breast_updated/breast/fastMRI_breast_IDS_281_290/fastMRI_breast_281_1.h5"
OUT = "manuscript/figures/figure5_examples.png"
os.makedirs("manuscript/figures", exist_ok=True)


def ifft2c(k):
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )


def coil_combine_rss(img_coils, axis=0):
    return np.sqrt(np.sum(np.abs(img_coils) ** 2, axis=axis))


def extract_phase_pca(img_coils):
    C, H, W = img_coils.shape
    flat = img_coils.reshape(C, -1)
    U, S, Vh = np.linalg.svd(flat, full_matrices=False)
    weights = U[:, 0].conj()
    combined = (weights[:, None] * flat).sum(0).reshape(H, W)
    phase = np.angle(combined)
    phase = np.unwrap(phase, axis=-1)
    return np.angle(np.exp(1j * phase))


def normalize(x, lo_pct=1, hi_pct=99):
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


# ---------------- Prostate ----------------
def load_prostate(path):
    with h5py.File(path, "r") as f:
        k = f["kspace"]            # (S, D, C, H, W) complex64
        S, D, C, H, W = k.shape
        sidx = S // 2 + 5
        # Use direction 0 only (cleaner phase than averaging all 34 directions)
        sl = np.array(k[sidx, 0])  # (C, H, W) complex
        img = ifft2c(sl)
    mag = coil_combine_rss(img)
    pha = extract_phase_pca(img)
    # rotate to upright
    mag = np.rot90(mag, k=1)
    pha = np.rot90(pha, k=1)
    return mag, pha, sidx


# ---------------- Breast ----------------
def load_breast(path):
    """Magnitude from temptv (clean reconstructed image),
    phase from raw k-space with zero-filled Cartesian assumption."""
    with h5py.File(path, "r") as f:
        # 1) Magnitude from temptv: (192 slices, 4 frames, 320, 320) float64
        temptv = f["temptv"]
        T = np.array(temptv)
        sidx_img = T.shape[0] // 2          # middle slice
        mag = T[sidx_img, 0]                # first dynamic frame
        # 2) Phase from raw k-space: (2, 288, 640, 16, 90) float64
        k = f["kspace"]
        K = k.shape                          # (2, S, kx, C, ky)
        sidx_k = K[1] // 2
        sl = np.array(k[:, sidx_k])         # (2, kx, C, ky)
        comp = sl[0] + 1j * sl[1]           # (kx, C, ky)
        comp = np.transpose(comp, (1, 0, 2))# (C, kx, ky)
        # Zero-fill ky from 90 to 640 (centered) so aspect ratio matches kx
        C, kx, ky = comp.shape
        kx_target = kx
        ky_target = kx
        pad_left = (ky_target - ky) // 2
        pad_right = ky_target - ky - pad_left
        comp_zf = np.pad(comp, ((0, 0), (0, 0), (pad_left, pad_right)),
                         mode="constant")
        img = ifft2c(comp_zf)               # (C, kx, kx) complex
        pha = extract_phase_pca(img)
        # crop center to ~320x320 to roughly match temptv resolution
        center = pha.shape[0] // 2
        crop = 160
        pha = pha[center - crop:center + crop, center - crop:center + crop]
        # match orientation/sizing of mag
        from scipy.ndimage import zoom as zoom_  # noqa: WPS433
        zy = mag.shape[0] / pha.shape[0]
        zx = mag.shape[1] / pha.shape[1]
        pha = zoom_(pha, (zy, zx), order=1)
    return mag, pha, (sidx_img, sidx_k)


def render(mag_p, pha_p, mag_b, pha_b, out_path):
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                          "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.6))

    rows = [
        ("Prostate (DWI)",       mag_p, pha_p),
        ("Breast (DCE radial)",  mag_b, pha_b),
    ]
    for r, (label, mag, pha) in enumerate(rows):
        m = normalize(mag)
        p_disp = (pha + np.pi) / (2 * np.pi)
        diff = p_disp - m

        ax = axes[r, 0]
        ax.imshow(m, cmap="gray", aspect="equal")
        ax.set_title(f"{label} \u2014 magnitude", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[r, 1]
        ax.imshow(p_disp, cmap="twilight", aspect="equal")
        ax.set_title(f"{label} \u2014 phase", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[r, 2]
        vmax = max(abs(diff.min()), abs(diff.max()))
        ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title("Phase \u2212 magnitude (normalized)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)


if __name__ == "__main__":
    print("Prostate file:", PROST_PATH)
    mag_p, pha_p, sidx_p = load_prostate(PROST_PATH)
    print(f"  prostate slice {sidx_p}, mag {mag_p.shape}")

    print("Breast file:", BREAST_PATH)
    mag_b, pha_b, sidxs = load_breast(BREAST_PATH)
    print(f"  breast slice (img {sidxs[0]}, kspace {sidxs[1]}), "
          f"mag {mag_b.shape}, phase {pha_b.shape}")

    render(mag_p, pha_p, mag_b, pha_b, OUT)
    print("Done.")
