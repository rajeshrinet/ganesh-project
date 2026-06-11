import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, Image, Video
import time
import matplotlib.patches as patches

# ==============================================================================
# 1. Compile the Cython module
# ==============================================================================
print("Compiling Cython module...")
result = subprocess.run(
    ["python", "Unbounded_setup.py", "build_ext", "--inplace"],
    capture_output=True, text=True
)

if result.returncode != 0:
    print("COMPILATION FAILED:")
    print(result.stderr[-2000:])
    raise RuntimeError("Cython compilation failed.")
else:
    print("Compilation successful.")

import UnboundedFilaments as filaments

# ==============================================================================
# 2. Parameters
# ==============================================================================
particle = 32
N = particle
b = 1.0
dt_base = 0.01
epsilon = 3
eta = 1.0 / 6
h = 1000 * b
periodicity = 0
bondLength = 2.1 * b
xl = 2 * b * particle * 1.1
lje = 1
ljr = 2.1 * b
wlje = 10
wljr = 2.1 * b

k_spring = 25
bending_strength = 13.125

# ==============================================================================
# 3. Activity modes
#    A = |vs| * N / (bending_strength * eta)
# ==============================================================================
activity_modes = {
    "corkscrew": -0.7355,   # A ≈ 10.76
    "beating":   -1.1032,   # A ≈ 16.14
    "aperiodic": -2.1334,   # A ≈ 31.21
}

print("\nActivity number verification:")
print(f"{'Mode':<12} {'vs':>8}  {'A (computed)':>14}  {'A (target)':>12}")
print("-" * 52)
targets = {"corkscrew": 10.76, "beating": 16.14, "aperiodic": 31.21}
for mode, vs in activity_modes.items():
    A_computed = abs(vs) * N / (bending_strength * eta)
    print(f"{mode:<12} {vs:>8.4f}  {A_computed:>14.2f}  {targets[mode]:>12.2f}")
print()

# ==============================================================================
# 4. Initial position
# ==============================================================================
def get_initial_position(seed=42):
    np.random.seed(seed)
    x_init = np.zeros(N)
    y_init = np.zeros(N)
    z_init = 2.1 + np.arange(N) * bondLength
    x_init[2:N] += 0.15 * np.random.random(N - 2)
    y_init[2:N] += 0.15 * np.random.random(N - 2)
    return np.concatenate((x_init, y_init, z_init))

# ==============================================================================
# 5. Simulation + static plot + animation
# ==============================================================================
T_total    = 500
dd         = 300
t_eval     = np.linspace(0, T_total, dd)
Nframe_map     = {"corkscrew": 50, "beating": 50, "aperiodic": 50}
delay_time_map = {"corkscrew": 400, "beating": 400, "aperiodic": 400}
n_snaps_map    = {"corkscrew": 30,  "beating": 50,  "aperiodic": 30}
cut_map        = {"corkscrew": 0,   "beating": int(0.4 * dd), "aperiodic": int(0.4 * dd)}

for mode_name, vs in activity_modes.items():
    print(f"\n{'='*50}")
    print(f"Running simulation for mode: {mode_name.upper()}  (vs = {vs},  "
          f"A = {abs(vs)*N/(bending_strength*eta):.2f})")

    Nframe     = Nframe_map[mode_name]
    delay_time = delay_time_map[mode_name]
    facT       = int(dd / Nframe)
    n_snaps    = n_snaps_map[mode_name]
    cut        = cut_map[mode_name]

    sim = filaments.colloids(
        N, 0, b, eta, k_spring, vs, bending_strength, bondLength, lje, ljr
    )
    position0 = get_initial_position(seed=42)

    def rhs(t, position):
        sim.computeVel_3tClamp(position)
        return np.array(sim.VV)

    t1  = time.perf_counter()
    sol = solve_ivp(
        rhs, [0, T_total], position0,
        method='LSODA', t_eval=t_eval,
        rtol=1e-4, atol=1e-4
    )
    t2  = time.perf_counter()

    X = sol.y.T
    print(f"Integration finished in {t2 - t1:.2f} s  |  "
          f"steps = {sol.t.size}  |  success = {sol.success}")

    tx = X[:, N - 1]
    ty = X[:, 2*N - 1]
    tz = X[:, 3*N - 1]
    print(f"Tip x range: [{tx.min():.3f}, {tx.max():.3f}]  "
          f"y range: [{ty.min():.3f}, {ty.max():.3f}]  "
          f"z range: [{tz.min():.3f}, {tz.max():.3f}]")

    # =========================================================================
    # A. Static plot — overlaid 3D snapshots + tip trajectory
    # =========================================================================
    print(f"Generating static plot for {mode_name}...")

    snap_idx = np.linspace(cut, dd - 1, n_snaps, dtype=int)

    fig_static = plt.figure(figsize=(7, 7))
    ax_s = fig_static.add_subplot(111, projection='3d')

    ax_s.plot(tx[cut:], ty[cut:], tz[cut:],
              color='black', lw=0.8, alpha=0.7, zorder=1)

    for idx in snap_idx:
        xb = X[idx, 0:N]
        yb = X[idx, N:2*N]
        zb = X[idx, 2*N:3*N]

        ax_s.plot(xb, yb, zb, '-', color='teal', lw=0.7, alpha=0.25)
        ax_s.scatter(xb[:-1], yb[:-1], zb[:-1],
                     color='teal', s=18, alpha=0.25,
                     depthshade=True, zorder=3)
        ax_s.scatter([xb[-1]], [yb[-1]], [zb[-1]],
                     color='maroon', s=40, alpha=1.0,
                     depthshade=False, zorder=4)

    ax_s.set_xlim(-30, 30)
    ax_s.set_ylim(-30, 30)
    ax_s.set_zlim(0, 70)
    ax_s.set_xlabel('x/b')
    ax_s.set_ylabel('y/b')
    ax_s.set_zlabel('z/b')
    ax_s.set_title(
        f'{mode_name.capitalize()} Mode  '
        f'($v_s$ = {vs},  $\\mathcal{{A}}$ = {abs(vs)*N/(bending_strength*eta):.2f})',
        fontsize=11
    )
    plt.tight_layout()

    static_filename = f'fig_3d_{mode_name}.png'
    fig_static.savefig(static_filename, dpi=150, bbox_inches='tight')
    plt.close(fig_static)

    display(Image(filename=static_filename))
    try:
        from google.colab import files
        files.download(static_filename)
    except ImportError:
        pass

    # =========================================================================
    # B. Animation
    # =========================================================================
    print(f"Generating animation for {mode_name}...")
    fig_anim = plt.figure(figsize=(5, 5))

    def animation(Nframe, delay_time):

        def prog(i):
            fig_anim.clf()
            ax = fig_anim.add_subplot(111, projection='3d')
            print(i, end=' ')

            x = X[i * facT, 0:N]
            y = X[i * facT, N:2*N]
            z = X[i * facT, 2*N:3*N]

            ax.plot(x, y, z, '-', color='teal', lw=1.5, alpha=0.8)
            ax.scatter(x, y, z, color='teal', s=30, zorder=5)
            ax.scatter([x[-1]], [y[-1]], [z[-1]], color='maroon', s=60, zorder=6)

            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)
            ax.set_zlim(0, 70)
            ax.set_xlabel('x/b')
            ax.set_ylabel('y/b')
            ax.set_zlabel('z/b')
            ax.set_title(
                f'$\\tau$={t_eval[i * facT]:1.2E}  |  {mode_name}  '
                f'($\\mathcal{{A}}$={abs(vs)*N/(bending_strength*eta):.2f})'
            )

        anm = FuncAnimation(fig_anim, prog, frames=Nframe,
                            interval=delay_time, repeat=False)
        return anm

    anim = animation(Nframe, delay_time)

    anim_filename = f'output_3d_{mode_name}.mp4'
    anim.save(anim_filename, writer='ffmpeg')
    fig_anim.clear()
    plt.close(fig_anim)
    print()

    try:
        display(Video(anim_filename, embed=True, width=600))
    except Exception:
        video_html = (
            f'<video width="600" controls>'
            f'<source src="{anim_filename}" type="video/mp4">'
            f'Your browser does not support the video tag.'
            f'</video>'
        )
        display(HTML(video_html))

    try:
        from google.colab import files
        files.download(anim_filename)
        print(f"-> Download initiated for {anim_filename}")
    except ImportError:
        pass

print("\nAll 3D plots, animations, and downloads completed successfully!")