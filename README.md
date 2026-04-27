**Identity-Conditioned Forgetting in Episodic Agent Memory**

A CNN-based episodic memory architecture for agents, where memories are stored as temporally ordered events and forgotten through identity-conditioned feature-space decay.

This repository contains a research prototype for studying selective forgetting in agent memory systems. Instead of storing all experiences indefinitely, the agent retains and forgets memories as a function of:

1. time
2. feature-level degradation
3. alignment with agent identity (self.facts)

The core idea is simple:

the agent does not forget images — it forgets features, and preferentially retains those aligned with what it believes it is.

**Overview**
Modern agent memory systems typically rely on one of two strategies:

Store everything (vector databases / retrieval memory)
Forget by recency (TTL, windowing, truncation)

Both are crude.
**This project explores a third alternative:**

**selective forgetting in feature space**

Memories are encoded as CNN feature embeddings and stored in an episodic memory structure. As time passes, these representations decay. The rate of forgetting is not uniform — it is conditioned on alignment with the agent’s internal identity representation (self.facts).

This creates a memory system that:

preserves chronology
stores outcomes
forgets continuously
retains self-relevant experiences longer
Core Idea

**Each experience is represented as an event node in memory.**

Every node contains:

a temporal position (when it occurred)
a CNN feature representation (what was perceived)
outcome branches (what happened)
an alignment score with agent identity (whether it matters)

Memory is organized as:

a linked list over time (episodic chronology)
a branching structure per event (success / failure outcomes)

**This gives memory two properties:**

temporal in storage
hierarchical in meaning
Architecture
**1. Episodic Memory (Linked List)**

Experiences are stored in temporal order:

event_1 → event_2 → event_3 → ... → event_t

This preserves chronology and enables event-local aging.

**2. Event Representation (CNN Features)**

Each event is encoded using a CNN backbone (ResNet18 in the prototype).

Raw images are not stored as memory.

Instead, memory stores:

feature embeddings
perceptual structure
compressed representations of experience

This allows forgetting to happen in feature space, not pixel space.

**3. Outcome Branches (Success / Failure Trees)**

Each event contains two branches:

success branch
failure branch

These branches represent outcome-conditioned memory traces.

This allows memory to retain not just what was seen, but what happened.

**4. Identity Representation (self.facts)**

The agent maintains an internal identity representation called self.facts.

This is a latent embedding representing what the agent is aligned with.

In the prototype, self.facts is constructed from class-conditioned CNN embeddings (e.g. vehicles).

In a real system, this could be:

user preferences
agent goals
stable beliefs
persistent facts
long-term reward priors
5. Identity-Conditioned Forgetting

**Memories decay over time in feature space.**

The forgetting rule is:

strength=e
−λ(1−alignment)t

where:

t = memory age
λ = base forgetting rate
alignment = cosine similarity with self.facts

This means:

high alignment → slower forgetting
low alignment → faster forgetting

The agent selectively preserves memories consistent with its identity.

Why This Matters

Most agent memory systems optimize retrieval.

This project studies retention.

The distinction matters.

A useful memory system should not only retrieve relevant experiences — it should also decide:

what should remain
what should degrade
what should disappear

This work treats forgetting as a first-class mechanism rather than a storage failure.

**Key Contributions**
1. Identity-Conditioned Forgetting

A memory decay mechanism where retention is conditioned on alignment with agent identity.

2. Feature-Space Forgetting

Memories decay in CNN representation space rather than through hard deletion or recency-only pruning.

3. Episodic Memory Structure

A linked-list memory architecture with event-local outcome branching.

4. Metric Critique

We show that naive forgetting metrics based on raw feature magnitude are structurally confounded and unsuitable for evaluating selective forgetting.

5. Normalized Retention

**Selective forgetting should be evaluated using retention ratio, not absolute feature magnitude.**

**Experimental Setup**

The prototype uses:

Dataset: CIFAR-10
Encoder: ResNet18 (pretrained)
Memory Type: linked-list episodic memory
Identity Prior: class-conditioned latent mean (self.facts)
Baselines:
no forgetting
pure time decay
random pruning
identity-conditioned forgetting
Experiments
1. Retrieval Under Forgetting

Tests whether memories remain retrievable as decay progresses.

Main result:
identity-conditioned forgetting preserves retrieval of self-relevant memories better than identity-agnostic baselines.

2. Forgetting Curves

Measures memory degradation over time under different forgetting rules.

3. Memory Selectivity

Measures whether identity-aligned memories are retained preferentially.

This experiment also exposes an important methodological result:

raw feature magnitude is a confounded metric for selective forgetting.

4. Lambda Ablation

Tests how forgetting aggressiveness changes retrieval and selectivity.

5. Identity Definition Ablation

Tests whether forgetting follows identity priors rather than fixed labels.

Main Result

Identity-conditioned forgetting preserves retrieval of self-relevant memories under decay while maintaining competitive overall retrieval performance.

Additionally, random forgetting exposes that raw feature magnitude is a misleading proxy for selective retention, motivating normalized retention as the correct evaluation metric.

Repository Structure
.
├── memory.py                # Linked-list episodic memory structures
├── encoder.py               # CNN feature encoder
├── forgetting.py            # Forgetting functions and decay rules
├── experiments.py           # Retrieval, selectivity, and ablation experiments
├── notebook.ipynb           # Google Colab research prototype
├── figures/                 # Plots for paper
└── README.md
Running the Prototype

This project is designed to run in Google Colab.

Open the notebook
Install dependencies
Run all cells
Generate:
forgetting curves
retrieval plots
selectivity plots
ablation results
Dependencies
Python 3.10+
PyTorch
Torchvision
NumPy
Matplotlib
scikit-learn
tqdm
Citation


Current focus:

normalized retention
stronger ablations
manuscript preparation
arXiv submission
License

**MIT**
