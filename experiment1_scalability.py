import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from stigmergy import Agent, SharedField

# ── Parameters ──────────────────────────────────────────────
DIM = 512
DELTA = 0.01      # convergence threshold
MAX_ROUNDS = 500
RUNS = 5          # runs per N for averaging

BETA = 0.0        # no observation step (scalability experiment)
ALPHA = 0.1       # same across all three protocols
THETA = 0.3
RHO = 0.1
ETA = 0.5

N_VALUES = [10, 20, 50, 100, 150, 200]


# ── Helpers ──────────────────────────────────────────────────
def make_concepts(n: int, dim: int) -> torch.Tensor:
    v = torch.randn(n, dim)
    return F.normalize(v, dim=1)


def max_belief_change(prev: list, curr: list) -> float:
    return max((p - c).norm().item() for p, c in zip(prev, curr))


# ── Stigmergy ─────────────────────────────────────────────────
def run_stigmergy(n: int, concepts: torch.Tensor) -> tuple[int, bool]:
    agents = [Agent(i, DIM, beta=BETA, alpha=ALPHA, theta=THETA) for i in range(n)]
    for i, agent in enumerate(agents):
        agent.belief = concepts[i].clone()

    field = SharedField(DIM, rho=RHO, eta=ETA)
    total_messages = 0

    for _ in range(MAX_ROUNDS):
        prev_beliefs = [a.belief.clone() for a in agents]

        field_state = field.read()
        field_zero = field.is_zero()
        deposits = []

        for i, agent in enumerate(agents):
            deposit = agent.step(concepts[i], field_state, field_zero)
            if deposit is not None:
                deposits.append(deposit)

        field.update(deposits)
        total_messages += len(deposits) + n

        # only check convergence after field has been read (round 1 skips field read)
        if not field_zero and max_belief_change(prev_beliefs, [a.belief for a in agents]) < DELTA:
            return total_messages, True

    return total_messages, False


# ── Star Topology ─────────────────────────────────────────────
def run_star(n: int, concepts: torch.Tensor) -> tuple[int, bool]:
    beliefs = [concepts[i].clone() for i in range(n)]
    total_messages = 0

    for _ in range(MAX_ROUNDS):
        prev_beliefs = [b.clone() for b in beliefs]

        hub_state = F.normalize(torch.stack(beliefs).mean(dim=0), dim=0)
        beliefs = [F.normalize((1 - ALPHA) * b + ALPHA * hub_state, dim=0) for b in beliefs]
        total_messages += 2 * n

        if max_belief_change(prev_beliefs, beliefs) < DELTA:
            return total_messages, True

    return total_messages, False


# ── Consensus ─────────────────────────────────────────────────
def run_consensus(n: int, concepts: torch.Tensor) -> tuple[int, bool]:
    beliefs = [concepts[i].clone() for i in range(n)]
    total_messages = 0

    for _ in range(MAX_ROUNDS):
        prev_beliefs = [b.clone() for b in beliefs]

        new_beliefs = []
        for i in range(n):
            others = torch.stack([beliefs[j] for j in range(n) if j != i])
            mean_other = F.normalize(others.mean(dim=0), dim=0)
            updated = (1 - ALPHA) * beliefs[i] + ALPHA * mean_other
            new_beliefs.append(F.normalize(updated, dim=0))
        beliefs = new_beliefs
        total_messages += n * (n - 1)

        if max_belief_change(prev_beliefs, beliefs) < DELTA:
            return total_messages, True

    return total_messages, False


# ── Main ─────────────────────────────────────────────────────
def main():
    results = {"stigmergy": [], "star": [], "consensus": []}
    converged = {"stigmergy": [], "star": [], "consensus": []}

    for n in N_VALUES:
        for key, fn in [("stigmergy", run_stigmergy),
                        ("star",      run_star),
                        ("consensus", run_consensus)]:
            msgs_list, conv_list = [], []
            for _ in range(RUNS):
                msgs, conv = fn(n, make_concepts(n, DIM))
                msgs_list.append(msgs)
                conv_list.append(conv)
            results[key].append(np.mean(msgs_list))
            converged[key].append(all(conv_list))

        print(
            f"N={n:3d} | stigmergy={results['stigmergy'][-1]:8.0f}"
            f" | star={results['star'][-1]:8.0f}"
            f" | consensus={results['consensus'][-1]:10.0f}"
            f"{'  [NOT CONVERGED]' if not converged['consensus'][-1] else ''}"
        )

    # ── Plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    styles = {
        "consensus": ("s-",  "Consensus",  "tab:blue"),
        "star":      ("^-",  "Star",       "tab:orange"),
        "stigmergy": ("o-",  "Stigmergy",  "tab:red"),
    }

    for key, (style, label, color) in styles.items():
        msgs = results[key]
        ax.plot(N_VALUES, msgs, style, label=label, color=color)

        # mark non-converged points with ×
        nc_x = [N_VALUES[i] for i, c in enumerate(converged[key]) if not c]
        nc_y = [msgs[i]     for i, c in enumerate(converged[key]) if not c]
        if nc_x:
            ax.scatter(nc_x, nc_y, marker="x", s=120, color="red", zorder=5,
                       label="not converged" if key == "consensus" else "")

    # reference lines for O(N) and O(N²) — anchored below data to stay visible
    n_arr = np.array(N_VALUES, dtype=float)
    ref_n  = n_arr  * (results["stigmergy"][0]  / N_VALUES[0]  * 0.3)
    ref_n2 = n_arr**2 * (results["consensus"][0] / N_VALUES[0]**2 * 0.3)
    ax.plot(N_VALUES, ref_n2, "k-.", linewidth=1.5, label="O(N²) ref")
    ax.plot(N_VALUES, ref_n,  "k:",  linewidth=1.5, label="O(N) ref")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_major_formatter(plt.LogFormatterMathtext())
    ax.set_xlabel("Number of agents N", fontsize=14)
    ax.set_ylabel("Total messages to convergence", fontsize=14)
    ax.set_title("")
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    plt.tight_layout()
    plt.savefig("figures/Fig_scalability.pdf", dpi=300)
    plt.savefig("figures/Fig_scalability.png", dpi=300)
    print("Saved figures/Fig_scalability.pdf/.png")
    plt.show()


if __name__ == "__main__":
    main()
