
import numpy as np
import matplotlib.pyplot as plt
import argparse


#####
import scipy as sp
import ot

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils
from pathlib import Path
## Data preparation
## The four distributions are constructed from 4 simple images (taken from POT tutorial)

def im2mat(img):
    """Converts and image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


this_file = Path(__file__).resolve()
data_path = this_file.parent / "simple_shapes"

square = plt.imread(data_path / "square.png").astype(np.float64)[:, :, 2]
cross = plt.imread(data_path / "cross.png").astype(np.float64)[:, :, 2]
triangle = plt.imread(data_path / "triangle.png").astype(np.float64)[:, :, 2]
star = plt.imread(data_path / "star.png").astype(np.float64)[:, :, 2]

shapes = [square, cross, triangle, star]

S = 4
xs = [[] for i in range(S)]

for nb in range(4):
    for i in range(8):
        for j in range(8):
            if shapes[nb][i, j] < 0.95:
                xs[nb].append([j, 8 - i])

xs = [np.array(xs[s]) for s in range(S)]


ns = [len(xs[s]) for s in range(S)]
""""Compute all distances matrices for the four shapes"""
Cs = [sp.spatial.distance.cdist(xs[s], xs[s]) for s in range(S)]
Cs = [cs / cs.max() for cs in Cs]

ps = [ot.unif(ns[s]) for s in range(S)]

i = 3

Cs_except_i = [C for j, C in enumerate(Cs) if j != i]
ps_except_i = [p for j, p in enumerate(ps) if j != i]


M = 30
p = ot.unif(M)


###################################################################################################

print('Estimating its GW-barycenter coordinates')

## FIXED-POINT METHOD
#obj_recon, lambdas_obj = utils.get_lambdas(Cs_except_i, ps_except_i,Cs[i],ps[i])
obj_recon, lambdas_obj = utils.get_lambdas_constraints(Cs_except_i, ps_except_i,Cs[i],ps[i])


##BLOW-UP METHOD
# obj_recon, b, temp_blow_up = utils.blow_up(Cs_except_i, ps_except_i,Cs[i],ps[i])
# obj_recon, lambdas_obj = utils.get_lambdas_blowup(temp_blow_up, obj_recon,b)

#####

special_pt = lambdas_obj

def plot_simplex_heatmap(pts, vals, special_pt=None):
    pts = np.asarray(pts)
    vals = np.asarray(vals).reshape(-1)

    # barycentric → 2D (equilateral triangle)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])

    xy = pts[:, 0, None] * A + pts[:, 1, None] * B + pts[:, 2, None] * C

    plt.figure(figsize=(6, 6))

    sc = plt.scatter(
        xy[:, 0],
        xy[:, 1],
        c=vals,
        s=80,
        cmap="plasma",  #inferno
        vmin=1e-3,
        vmax=0.01
    )

    plt.colorbar(sc)

    # draw simplex edges
    tri = np.array([A, B, C, A])
    plt.plot(tri[:, 0], tri[:, 1], 'k-')

    # optional black cross
    if special_pt is not None:
        special_pt = np.asarray(special_pt)
        sp_xy = special_pt[0] * A + special_pt[1] * B + special_pt[2] * C
        plt.scatter(sp_xy[0], sp_xy[1], c='k', marker='x', s=80)

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()






def main():
    parser = argparse.ArgumentParser(
        description="Plot a heatmap on the simplex from saved .npy files."
    )
    parser.add_argument(
        "--pts",
        type=str,
        default="simplex_grid.npy",
        help="Path to the .npy file with simplex points (N,3).",
    )
    parser.add_argument(
        "--vals",
        type=str,
        default="gw_distances_target.npy",
        help="Path to the .npy file with values (N,).",
    )
    parser.add_argument(
        "--special",
        type=float,
        nargs=3,
        default=None,
        metavar=("L1", "L2", "L3"),
        help="Optional special barycentric point (L1 L2 L3) to mark with a black cross.",
    )

    args = parser.parse_args()

    pts = np.load(args.pts)
    vals = np.load(args.vals)

    special_pt = lambdas_obj

    plot_simplex_heatmap(pts, vals, special_pt=special_pt)


if __name__ == "__main__":
    main()




from mpl_toolkits.mplot3d import Axes3D


# --- Load data ---
pts = np.load("simplex_grid.npy")        # shape (N, 3)
vals = np.load("gw_distances_target.npy")       # shape (N,)

# --- Extract first two simplex coordinates ---
x = pts[:, 0]
y = pts[:, 1]
z = vals

# --- Load or define special point (barycentric) ---
# Replace this with however you load lambdas_obj if needed
lambdas_obj = np.array([0.2, 0.5, 0.3])   # <-- PUT YOUR REAL VALUES HERE

x_sp = lambdas_obj[0]
y_sp = lambdas_obj[1]
z_sp = 0.0   # you can also set this to a GW value if desired

# --- 3D scatter plot ---
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    x, y, z,
    c=z,
    s=35
)

# --- Special point ---
ax.scatter(
    x_sp, y_sp, z_sp,
    c="k",
    marker="*",
    s=200,
    label="special point"
)

fig.colorbar(sc, ax=ax, shrink=0.6, label="GW distance")

ax.set_xlabel(r"$\lambda_1$")
ax.set_ylabel(r"$\lambda_2$")
ax.set_zlabel("GW distance")

ax.set_title("3D plot of GW distances over simplex grid")
ax.legend()

# Optional z-range
ax.set_zlim(0, 0.02)

plt.tight_layout()
plt.show()