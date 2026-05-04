import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from stigmergy import Agent, SharedField

# ── Parameters ──────────────────────────────────────────────
DIM = 512
DELTA = 0.001     # convergence threshold (strict for clean log N scaling)
MAX_ROUNDS = 500
RUNS = 5          # runs per N for averaging

BETA = 0.0        # no observation step (scalability experiment)
ALPHA = 0.5       # same across all three protocols (true averaging for gossip)
THETA = 0.3
RHO = 0.1
ETA = 0.5

N_VALUES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


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


# ── Gossip ────────────────────────────────────────────────────
def run_gossip(n: int, concepts: torch.Tensor) -> tuple[int, bool]:
    """Pairwise gossip (canonical): random pair matching per round.
    Each pair does a symmetric push-pull update with weight ALPHA.
    Messages: 2 per pair → N messages per round → O(N log N) total."""
    beliefs = [concepts[i].clone() for i in range(n)]
    total_messages = 0

    for _ in range(MAX_ROUNDS):
        prev_beliefs = [b.clone() for b in beliefs]

        perm = np.random.permutation(n)
        new_beliefs = [b.clone() for b in beliefs]
        n_pairs = n // 2

        for k in range(n_pairs):
            i, j = int(perm[2 * k]), int(perm[2 * k + 1])
            bi, bj = beliefs[i], beliefs[j]
            new_beliefs[i] = F.normalize((1 - ALPHA) * bi + ALPHA * bj, dim=0)
            new_beliefs[j] = F.normalize((1 - ALPHA) * bj + ALPHA * bi, dim=0)

        total_messages += 2 * n_pairs  # 2 messages per pair → N per round
        beliefs = new_beliefs

        if max_belief_change(prev_beliefs, beliefs) < DELTA:
            return total_messages, True

    return total_messages, False


# ── Gossip k=3 (push-pull, fixed neighbors) ──────────────────
def run_gossip_k3(n: int, concepts: torch.Tensor) -> tuple[int, bool]:
    """Push-pull gossip with K=3 fixed random neighbors per round.
    Each agent picks K partners; symmetric averaging on both sides.
    Messages: 2*K*N = 6N per round → O(N log N) total."""
    K = 3
    beliefs = concepts.clone()   # (N, D)
    total_messages = 0

    for _ in range(MAX_ROUNDS):
        prev = beliefs.clone()

        # Sample K random partners != self for each agent (vectorized)
        rand_idx = np.random.randint(0, n - 1, size=(n, K))
        row_idx  = np.arange(n)[:, None]
        rand_idx = np.where(rand_idx >= row_idx, rand_idx + 1, rand_idx)
        idx_t    = torch.from_numpy(rand_idx).long()

        # Pull side: i receives K partners' beliefs
        pull_sum   = prev[idx_t].sum(dim=1)             # (N, D)
        pull_count = K * torch.ones(n)

        # Push side: each j receives belief from every i that picked j
        push_sum   = torch.zeros_like(prev)
        push_count = torch.zeros(n)
        flat_idx   = idx_t.flatten()                                       # (N*K,)
        src_b      = prev.unsqueeze(1).expand(-1, K, -1).reshape(-1, prev.shape[-1])
        push_sum.index_add_(0, flat_idx, src_b)
        push_count.index_add_(0, flat_idx, torch.ones(n * K))

        avg = (pull_sum + push_sum) / (pull_count + push_count).unsqueeze(1)
        beliefs = F.normalize((1 - ALPHA) * prev + ALPHA * avg, dim=1)

        total_messages += 2 * K * n  # push-pull: 2 msgs per partnership, K per agent

        if (prev - beliefs).norm(dim=1).max().item() < DELTA:
            return total_messages, True

    return total_messages, False


# ── Consensus ─────────────────────────────────────────────────
def run_consensus(n: int, concepts: torch.Tensor) -> tuple[int, bool]:
    """Vectorized all-to-all consensus for speed at large N."""
    B = concepts.clone()   # (N, D)
    total_messages = 0

    for _ in range(MAX_ROUNDS):
        prev_B = B.clone()

        total = B.sum(dim=0, keepdim=True)          # (1, D)
        mean_others = (total - B) / (n - 1)          # (N, D)
        mean_others = F.normalize(mean_others, dim=1)
        B = F.normalize((1 - ALPHA) * B + ALPHA * mean_others, dim=1)
        total_messages += n * (n - 1)

        if (prev_B - B).norm(dim=1).max().item() < DELTA:
            return total_messages, True

    return total_messages, False


# ── Main ─────────────────────────────────────────────────────
def main():
    results = {"stigmergy": [], "gossip": [], "gossip_k3": [], "consensus": []}
    converged = {"stigmergy": [], "gossip": [], "gossip_k3": [], "consensus": []}

    for n in N_VALUES:
        for key, fn in [("stigmergy", run_stigmergy),
                        ("gossip",    run_gossip),
                        ("gossip_k3", run_gossip_k3),
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
            f" | gossip={results['gossip'][-1]:8.0f}"
            f" | gossip_k3={results['gossip_k3'][-1]:8.0f}"
            f" | consensus={results['consensus'][-1]:10.0f}"
            f"{'  [NOT CONVERGED]' if not converged['consensus'][-1] else ''}"
        )

    # ── Plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    styles = {
        "consensus": ("s-",  "Consensus",          "tab:blue",   2.0),
        "gossip_k3": ("D-",  "Gossip ($k$=3)",     "tab:green",  2.0),
        "gossip":    ("^--", "Gossip (pairwise)",  "tab:orange", 2.5),
        "stigmergy": ("o-",  "Stigmergy",          "tab:red",    2.5),
    }

    for key, (style, label, color, lw) in styles.items():
        msgs = results[key]
        per_agent = [m / n for m, n in zip(msgs, N_VALUES)]
        ax.plot(N_VALUES, per_agent, style, label=label, color=color,
                linewidth=lw, markersize=7, zorder=3)

        # mark non-converged points with ×
        nc_x = [N_VALUES[i] for i, c in enumerate(converged[key]) if not c]
        nc_y = [per_agent[i] for i, c in enumerate(converged[key]) if not c]
        if nc_x:
            ax.scatter(nc_x, nc_y, marker="x", s=120, color="red", zorder=5,
                       label="not converged" if key == "consensus" else "")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_major_formatter(plt.LogFormatterMathtext())
    ax.set_xlabel("Number of agents $N$", fontsize=14)
    ax.set_ylabel("Messages per agent to convergence", fontsize=14)
    ax.set_title("")
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12, loc="upper left")
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
