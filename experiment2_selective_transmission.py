import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from stigmergy import Agent, SharedField

# ── Parameters ──────────────────────────────────────────────
DIM = 512
ALPHA = 0.1
BETA = 0.45       # observation anchor (noise enabled)
SIGMA = 1.0       # observation noise
RHO = 0.1
ETA = 0.5
MAX_ROUNDS = 100
RUNS = 20
N = 50


def make_concepts(n):
    v = torch.randn(n, DIM)
    return F.normalize(v, dim=1)


def sample_obs(concept):
    noise = torch.randn_like(concept) * SIGMA
    return F.normalize(concept + noise, dim=0)


def mean_pairwise_l2(beliefs):
    """Mean pairwise L2 distance among all agent beliefs."""
    B = torch.stack(beliefs)   # (N, dim)
    n = B.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append((B[i] - B[j]).norm().item())
    return np.mean(dists)


def belief_coherence(beliefs):
    """Order parameter: norm of the mean normalized belief vector.
    0 = incoherent (random directions), 1 = perfectly aligned."""
    B = F.normalize(torch.stack(beliefs), dim=1)  # (N, dim)
    return B.mean(dim=0).norm().item()


# ── Alignment experiment (shared concept, panel c) ──────────
SIGMA_ALIGN = 0.0   # no noise so convergence reaches 1.0

def run_alignment(n, theta, concept):
    """All agents observe the same concept; track mean cosine sim to it."""
    agents = [Agent(i, DIM, beta=BETA, alpha=ALPHA, theta=theta) for i in range(n)]
    for agent in agents:
        agent.belief = F.normalize(torch.randn(DIM), dim=0)

    field = SharedField(DIM, rho=RHO, eta=ETA)
    alignments = []

    for _ in range(MAX_ROUNDS):
        field_state = field.read()
        field_zero = field.is_zero()
        deposits = []

        for agent in agents:
            noise = torch.randn_like(concept) * SIGMA_ALIGN
            obs = F.normalize(concept + noise, dim=0)
            deposit = agent.step(obs, field_state, field_zero)
            if deposit is not None:
                deposits.append(deposit)

        field.update(deposits)

        B = F.normalize(torch.stack([a.belief for a in agents]), dim=1)
        c_norm = F.normalize(concept.unsqueeze(0), dim=1)
        sims = F.cosine_similarity(B, c_norm.expand_as(B), dim=1)
        alignments.append(sims.mean().item())

    return alignments


def avg_alignment(n, theta, runs=RUNS):
    all_align = []
    for _ in range(runs):
        concept = F.normalize(torch.randn(DIM), dim=0)
        all_align.append(run_alignment(n, theta, concept))
    return np.mean(all_align, axis=0)


def run_gating(n, theta, concepts):
    """Returns (deposit_ratios, cumulative_deposits, belief_divergences, alignments) per round."""
    agents = [Agent(i, DIM, beta=BETA, alpha=ALPHA, theta=theta) for i in range(n)]
    for i, agent in enumerate(agents):
        agent.belief = concepts[i].clone()

    field = SharedField(DIM, rho=RHO, eta=ETA)
    deposit_ratios = []
    divergences = []
    alignments = []

    for _ in range(MAX_ROUNDS):
        field_state = field.read()
        field_zero = field.is_zero()
        deposits = []

        for i, agent in enumerate(agents):
            obs = sample_obs(concepts[i])
            deposit = agent.step(obs, field_state, field_zero)
            if deposit is not None:
                deposits.append(deposit)

        field.update(deposits)
        deposit_ratios.append(len(deposits) / n)
        divergences.append(mean_pairwise_l2([a.belief for a in agents]))
        alignments.append(belief_coherence([a.belief for a in agents]))

    cum_deposits = np.cumsum([r * n for r in deposit_ratios])
    return deposit_ratios, cum_deposits, divergences, alignments


def avg_runs(n, theta, runs=RUNS):
    all_ratios, all_cum, all_div, all_align = [], [], [], []
    for _ in range(runs):
        r, c, d, a = run_gating(n, theta, make_concepts(n))
        all_ratios.append(r)
        all_cum.append(c)
        all_div.append(d)
        all_align.append(a)
    return (np.mean(all_ratios, axis=0),
            np.mean(all_cum, axis=0),
            np.mean(all_div, axis=0),
            np.mean(all_align, axis=0))


# ── Main figure: 3 panels by θ ───────────────────────────────
def main():
    THETAS = [0.3, 0.6, 0.9]
    colors = ["tab:blue", "tab:orange", "tab:red"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    data = {}
    for theta in THETAS:
        data[theta] = avg_runs(N, theta)

    align_data = {}
    for theta in THETAS:
        align_data[theta] = avg_alignment(N, theta)

    rounds = np.arange(1, MAX_ROUNDS + 1)

    # ── Panel 1: Deposit rate ─────────────────────────────────
    smooth_window = 7

    def rolling_mean(arr, w):
        kernel = np.ones(w) / w
        out = np.convolve(arr, kernel, mode='same')
        half = w // 2
        for i in range(half):
            out[i] = np.mean(arr[:2 * i + 1])
            out[-(i + 1)] = np.mean(arr[-(2 * i + 1):])
        return out

    ax = axes[0]
    for theta, color in zip(THETAS, colors):
        ratios, _, _, _ = data[theta]
        smoothed = rolling_mean(ratios, smooth_window)
        ax.plot(rounds, smoothed, color=color, label=f"$\\theta$={theta}")
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Deposit rate", fontsize=11)
    ax.set_title("")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # ── Panel 2: Cumulative deposits (with end-of-line labels) ──
    ax = axes[1]
    label_vals = [5000, 1840, 69]
    # (round_x_position, y_offset, va) for each line
    label_configs = [
        (rounds[-1] - 3,  130, "bottom"), # blue  5,000 – above line, inside expanded ylim
        (rounds[-1] - 3,   60, "bottom"), # orange 1,840 – above line
        (rounds[-1] - 3,   30, "bottom"), # red      69  – above line
    ]
    for theta, color, val, (rx, voff, va) in zip(THETAS, colors, label_vals, label_configs):
        _, cum, _, _ = data[theta]
        ax.plot(rounds, cum, color=color, label=f"$\\theta$={theta}")
        ax.text(rx, cum[-1] + voff,
                f"{val:,}",
                color="black", ha="right", va=va,
                fontsize=9, fontweight="bold")

    ax.set_ylim(0, 5600)   # extra headroom so 5,000 label sits inside the plot
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Cumulative deposits", fontsize=11)
    ax.set_title("")
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # ── Panel 3: Belief alignment across θ (shared concept) ──
    ax = axes[2]
    for theta, color in zip(THETAS, colors):
        ax.plot(rounds, align_data[theta], color=color, label=f"$\\theta$={theta}")

    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Cosine similarity to concept", fontsize=11)
    ax.set_title("")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # ── Sub-figure labels below each panel ───────────────────
    for ax, label in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.text(0.5, -0.22, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=11)

    plt.tight_layout(pad=1.0, w_pad=3.0, h_pad=2.0)
    plt.savefig("figures/Fig_gating_theta.pdf", dpi=300)
    plt.savefig("figures/Fig_gating_theta.png", dpi=300)
    print("Saved figures/Fig_gating_theta.pdf/.png")
    plt.show()


if __name__ == "__main__":
    main()
