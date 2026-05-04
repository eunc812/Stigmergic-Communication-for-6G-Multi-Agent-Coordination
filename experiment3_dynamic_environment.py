import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from stigmergy import Agent, SharedField

# ── Parameters ──────────────────────────────────────────────
DIM = 512
N = 30
K = 5           # early adopters
ALPHA = K / N   # field reading weight = K/N ≈ 0.167
BETA = 0.3      # observation weight (adopters only)
SIGMA = 0.05    # observation noise
RHO = 0.15      # field evaporation rate (panel 1)
RHO_HIGH = 0.9   # higher decay rate for panel 2 comparison
ETA = 0.1       # deposit weight
THETA = 0.3     # gating threshold
WARMUP_ROUNDS = 50
SHIFT_INTERVAL = 50   # rounds per concept phase
N_PHASES = 4          # A→B→C→D (4 shifts)
TOTAL_ROUNDS = SHIFT_INTERVAL * N_PHASES   # 200 rounds
RUNS = 20


def make_concept():
    v = torch.randn(DIM)
    return F.normalize(v, dim=0)


def sample_obs(concept):
    noise = torch.randn_like(concept) * SIGMA
    return F.normalize(concept + noise, dim=0)


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ── Panel 1: basic field adaptation ──────────────────────────
def run_experiment():
    concept_A = make_concept()
    concept_B = -concept_A  # antipodal

    adopter_beliefs = [concept_A.clone() for _ in range(K)]
    passive_beliefs = [concept_A.clone() for _ in range(N - K)]
    field = SharedField(DIM, rho=RHO, eta=ETA)

    field_sim_B = []
    passive_sim_B = []

    for _ in range(SHIFT_INTERVAL):   # show first 50 rounds
        field_state = field.read()
        field_zero = field.is_zero()
        deposits = []

        for idx in range(K):
            obs = sample_obs(concept_B)
            adopter_beliefs[idx] = (1 - BETA) * adopter_beliefs[idx] + BETA * obs
            norm = adopter_beliefs[idx].norm()
            if norm > 0:
                adopter_beliefs[idx] = adopter_beliefs[idx] / norm
            deposits.append(adopter_beliefs[idx].clone())

        for idx in range(N - K):
            if not field_zero:
                passive_beliefs[idx] = (1 - ALPHA) * passive_beliefs[idx] + ALPHA * field_state
                norm = passive_beliefs[idx].norm()
                if norm > 0:
                    passive_beliefs[idx] = passive_beliefs[idx] / norm

        field.update(deposits)

        if field.is_zero():
            field_sim_B.append(0.0)
        else:
            field_sim_B.append(cosine_sim(field.state, concept_B))

        p_sims = [cosine_sim(b, concept_B) for b in passive_beliefs]
        passive_sim_B.append(np.mean(p_sims))

    return np.array(field_sim_B), np.array(passive_sim_B)


# ── Panel 2: multi-shift stigmergy ───────────────────────────
def _warmup(adopter_beliefs, passive_beliefs, field, concept):
    for _ in range(WARMUP_ROUNDS):
        deposits = []
        for idx in range(K):
            obs = sample_obs(concept)
            adopter_beliefs[idx] = (1 - BETA) * adopter_beliefs[idx] + BETA * obs
            norm = adopter_beliefs[idx].norm()
            if norm > 0:
                adopter_beliefs[idx] = adopter_beliefs[idx] / norm
            deposits.append(adopter_beliefs[idx].clone())
        field_state = field.read()
        field_zero = field.is_zero()
        for idx in range(N - K):
            if not field_zero:
                passive_beliefs[idx] = (1 - ALPHA) * passive_beliefs[idx] + ALPHA * field_state
                norm = passive_beliefs[idx].norm()
                if norm > 0:
                    passive_beliefs[idx] = passive_beliefs[idx] / norm
        field.update(deposits)


def _make_2d_basis():
    """Two orthogonal unit vectors in DIM-space (Gram-Schmidt)."""
    e1 = F.normalize(torch.randn(DIM), dim=0)
    v  = torch.randn(DIM)
    v  = v - torch.dot(v, e1) * e1
    e2 = F.normalize(v, dim=0)
    return e1, e2


def _rotation_concepts(e1, e2):
    """Warmup concept + N_PHASES phase concepts, 90° apart in the e1-e2 plane.
    warmup=e1, phases=[e2, -e1, -e2, e1, ...]"""
    basis = [e1, e2, -e1, -e2]
    return e1, [basis[(i + 1) % 4] for i in range(N_PHASES)]


def run_multi_shift_stigmergy(rho):
    """Concepts rotate 90° every SHIFT_INTERVAL rounds in a fixed 2D subspace.
    rho=0: field accumulates all past directions → average drifts toward centre → lag grows.
    rho=RHO_HIGH: old signal evaporates → field tracks current concept consistently."""
    e1, e2 = _make_2d_basis()
    warmup_concept, phase_concepts = _rotation_concepts(e1, e2)

    adopter_beliefs = [warmup_concept.clone() for _ in range(K)]
    passive_beliefs = [warmup_concept.clone() for _ in range(N - K)]
    field = SharedField(DIM, rho=rho, eta=ETA)

    _warmup(adopter_beliefs, passive_beliefs, field, warmup_concept)

    sims = []
    for phase in range(N_PHASES):
        current = phase_concepts[phase]
        for _ in range(SHIFT_INTERVAL):
            field_state = field.read()
            field_zero  = field.is_zero()
            deposits = []

            for idx in range(K):
                obs = sample_obs(current)
                adopter_beliefs[idx] = (1 - BETA) * adopter_beliefs[idx] + BETA * obs
                norm = adopter_beliefs[idx].norm()
                if norm > 0:
                    adopter_beliefs[idx] = adopter_beliefs[idx] / norm
                deposits.append(adopter_beliefs[idx].clone())

            for idx in range(N - K):
                if not field_zero:
                    passive_beliefs[idx] = (1 - ALPHA) * passive_beliefs[idx] + ALPHA * field_state
                    norm = passive_beliefs[idx].norm()
                    if norm > 0:
                        passive_beliefs[idx] = passive_beliefs[idx] / norm

            field.update(deposits)

            all_sims = ([cosine_sim(b, current) for b in adopter_beliefs] +
                        [cosine_sim(b, current) for b in passive_beliefs])
            sims.append(np.mean(all_sims))

    return np.array(sims)


def run_multi_shift_consensus():
    """Consensus baseline with 90° rotating concepts.
    Passives blend toward mean of all N agents."""
    e1, e2 = _make_2d_basis()
    warmup_concept, phase_concepts = _rotation_concepts(e1, e2)

    adopter_beliefs = [warmup_concept.clone() for _ in range(K)]
    passive_beliefs = [warmup_concept.clone() for _ in range(N - K)]

    # Warmup
    for _ in range(WARMUP_ROUNDS):
        for idx in range(K):
            obs = sample_obs(warmup_concept)
            adopter_beliefs[idx] = (1 - BETA) * adopter_beliefs[idx] + BETA * obs
            norm = adopter_beliefs[idx].norm()
            if norm > 0:
                adopter_beliefs[idx] = adopter_beliefs[idx] / norm
        all_mean = torch.mean(torch.stack(adopter_beliefs + passive_beliefs), dim=0)
        mean_norm = all_mean.norm()
        if mean_norm > 1e-6:
            all_mean = all_mean / mean_norm
            for idx in range(N - K):
                passive_beliefs[idx] = (1 - ALPHA) * passive_beliefs[idx] + ALPHA * all_mean
                norm = passive_beliefs[idx].norm()
                if norm > 0:
                    passive_beliefs[idx] = passive_beliefs[idx] / norm

    sims = []
    for phase in range(N_PHASES):
        current = phase_concepts[phase]
        for _ in range(SHIFT_INTERVAL):
            for idx in range(K):
                obs = sample_obs(current)
                adopter_beliefs[idx] = (1 - BETA) * adopter_beliefs[idx] + BETA * obs
                norm = adopter_beliefs[idx].norm()
                if norm > 0:
                    adopter_beliefs[idx] = adopter_beliefs[idx] / norm
            all_mean = torch.mean(torch.stack(adopter_beliefs + passive_beliefs), dim=0)
            mean_norm = all_mean.norm()
            if mean_norm > 1e-6:
                all_mean = all_mean / mean_norm
                for idx in range(N - K):
                    passive_beliefs[idx] = (1 - ALPHA) * passive_beliefs[idx] + ALPHA * all_mean
                    norm = passive_beliefs[idx].norm()
                    if norm > 0:
                        passive_beliefs[idx] = passive_beliefs[idx] / norm
            all_sims = ([cosine_sim(b, current) for b in adopter_beliefs] +
                        [cosine_sim(b, current) for b in passive_beliefs])
            sims.append(np.mean(all_sims))

    return np.array(sims)


def avg_runs():
    all_field, all_passive = [], []
    all_hd, all_nd, all_cons = [], [], []
    for _ in range(RUNS):
        f, p = run_experiment()
        hd = run_multi_shift_stigmergy(rho=RHO_HIGH)
        nd = run_multi_shift_stigmergy(rho=0.0)
        c  = run_multi_shift_consensus()
        all_field.append(f)
        all_passive.append(p)
        all_hd.append(hd)
        all_nd.append(nd)
        all_cons.append(c)
    return (np.mean(all_field, axis=0),  np.std(all_field, axis=0),
            np.mean(all_passive, axis=0), np.std(all_passive, axis=0),
            np.mean(all_hd, axis=0),      np.std(all_hd, axis=0),
            np.mean(all_nd, axis=0),      np.std(all_nd, axis=0),
            np.mean(all_cons, axis=0),    np.std(all_cons, axis=0))


def main():
    print("Running belief spread experiment...")
    (f_mean, f_std, p_mean, p_std,
     hd_mean, hd_std,
     nd_mean, nd_std,
     c_mean,  c_std) = avg_runs()

    fig, ax = plt.subplots(figsize=(6, 4))

    rounds_p2 = np.arange(1, TOTAL_ROUNDS + 1)

    # ── Repeated concept shifts ──────────────────────────────
    ax.plot(rounds_p2, hd_mean, color="tab:green",
            label=f"Stigmergy w/ decay ($\\rho$={RHO_HIGH})")
    ax.fill_between(rounds_p2, hd_mean - hd_std, hd_mean + hd_std,
                    color="tab:green", alpha=0.2)
    ax.plot(rounds_p2, nd_mean, color="tab:orange",
            label="Stigmergy w/o decay ($\\rho$=0)")
    ax.fill_between(rounds_p2, nd_mean - nd_std, nd_mean + nd_std,
                    color="tab:orange", alpha=0.2)
    ax.plot(rounds_p2, c_mean, color="tab:red",
            label="Consensus")
    ax.fill_between(rounds_p2, c_mean - c_std, c_mean + c_std,
                    color="tab:red", alpha=0.15)

    # vertical lines at each concept shift
    for i in range(1, N_PHASES):
        ax.axvline(i * SHIFT_INTERVAL, color="black", linewidth=0.7,
                   linestyle=":", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Round", fontsize=9)
    ax.set_ylabel("Mean agent cosine similarity to current concept", fontsize=9)
    ax.set_title("")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8.5)
    ax.grid(False)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout(pad=1.0)
    plt.savefig("figures/Fig_belief_spread.pdf", dpi=300)
    plt.savefig("figures/Fig_belief_spread.png", dpi=300)
    print("Saved figures/Fig_belief_spread.pdf/.png")
    plt.show()


if __name__ == "__main__":
    main()
