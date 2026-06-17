# ============================================================
# CNN-Based Identity-Aligned Forgetting Memory
# Google Colab Ready Prototype
# ============================================================
# Core idea:
# - Each event is a node in a linked list.
# - Each event stores CNN feature representations.
# - Each event has success/failure branches.
# - Forgetting happens in CNN feature space using exponential decay.
# - Decay is modulated by alignment with self.facts.
# - Tests compare identity-aligned forgetting against baselines.
# ============================================================

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# ============================================================
# 2. Dataset
# ============================================================
# We use CIFAR-10 for a simple paper-style demo.
# You can replace this with your own agent experience images later.

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

class_names = train_dataset.classes
print(class_names)


# ============================================================
# 3. CNN Feature Extractor
# ============================================================
# We use a pretrained ResNet18 as the CNN encoder.
# The classifier head is removed.
# Output shape: [batch_size, 512]

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        features = self.backbone(x)
        features = features.flatten(1)
        features = F.normalize(features, p=2, dim=1)
        return features


encoder = CNNFeatureExtractor().to(DEVICE)
encoder.eval()


@torch.no_grad()
def encode_batch(images):
    images = images.to(DEVICE)
    return encoder(images).detach().cpu()


@torch.no_grad()
def encode_single(image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    return encoder(image_tensor).squeeze(0).detach().cpu()


# ============================================================
# 4. Memory Data Structures
# ============================================================

@dataclass
class OutcomeBranch:
    """
    Stores successful or failed visual experiences for one event.
    In a larger system, this can be a real tree.
    Here, each branch stores feature embeddings and metadata.
    """
    branch_type: str
    features: List[torch.Tensor] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)

    def add(self, feature: torch.Tensor, label: int, timestamp: int):
        self.features.append(feature.clone())
        self.labels.append(label)
        self.timestamps.append(timestamp)

    def mean_feature(self):
        if len(self.features) == 0:
            return None
        return torch.stack(self.features).mean(dim=0)


@dataclass
class MemoryNode:
    """
    Linked-list node.
    Each node is an event.
    key_feature represents the event.
    value contains success and failure branches.
    """
    event_id: int
    key_feature: torch.Tensor
    label: int
    timestamp: int
    success_branch: OutcomeBranch = field(default_factory=lambda: OutcomeBranch("success"))
    failure_branch: OutcomeBranch = field(default_factory=lambda: OutcomeBranch("failure"))
    next: Optional["MemoryNode"] = None

    original_strength: float = 1.0
    current_strength: float = 1.0
    alignment_score: float = 0.0


class LinkedListMemory:
    def __init__(self):
        self.head = None
        self.tail = None
        self.nodes = []

    def append(self, node: MemoryNode):
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self.nodes.append(node)

    def __len__(self):
        return len(self.nodes)

    def iter_nodes(self):
        current = self.head
        while current is not None:
            yield current
            current = current.next


# ============================================================
# 5. Agent Identity: self.facts
# ============================================================
# For the prototype, self.facts is represented as the average CNN embedding
# of classes the agent considers identity-aligned.
# Example: the agent cares about vehicles: automobile, ship, truck.

IDENTITY_CLASSES = ["automobile", "ship", "truck"]
IDENTITY_CLASS_IDS = [class_names.index(c) for c in IDENTITY_CLASSES]
print("Identity-aligned classes:", IDENTITY_CLASSES, IDENTITY_CLASS_IDS)


def build_self_facts(dataset, class_ids, samples_per_class=50):
    selected_indices = []
    class_counts = {cid: 0 for cid in class_ids}

    for idx, (_, label) in enumerate(dataset):
        if label in class_ids and class_counts[label] < samples_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
        if all(count >= samples_per_class for count in class_counts.values()):
            break

    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)

    all_features = []
    for images, labels in tqdm(loader, desc="Building self.facts embedding"):
        feats = encode_batch(images)
        all_features.append(feats)

    self_facts = torch.cat(all_features, dim=0).mean(dim=0)
    self_facts = F.normalize(self_facts, p=2, dim=0)
    return self_facts


self_facts = build_self_facts(train_dataset, IDENTITY_CLASS_IDS)
print("self.facts shape:", self_facts.shape)


# ============================================================
# 6. Alignment Function
# ============================================================

def cosine_alignment(feature: torch.Tensor, self_facts: torch.Tensor) -> float:
    feature = F.normalize(feature, p=2, dim=0)
    self_facts = F.normalize(self_facts, p=2, dim=0)
    score = torch.dot(feature, self_facts).item()
    # Map cosine from [-1, 1] to [0, 1] for easier decay control.
    return (score + 1.0) / 2.0


# ============================================================
# 7. Forgetting Functions
# ============================================================

def identity_aligned_decay(
    feature: torch.Tensor,
    t: float,
    alignment_score: float,
    base_lambda: float = 0.08,
    noise_scale: float = 0.03,
    prune_threshold: float = 0.01
):
    """
    CNN feature-space forgetting.

    Higher alignment -> slower forgetting.
    Lower alignment -> faster forgetting.

    Formula:
        strength = exp(-lambda * (1 - alignment) * t)

    Then:
        decayed_feature = feature * strength + noise
        prune tiny values
        re-normalize
    """
    strength = math.exp(-base_lambda * (1.0 - alignment_score) * t)

    decayed = feature * strength

    noise = torch.randn_like(decayed) * noise_scale * (1.0 - strength)
    decayed = decayed + noise

    mask = decayed.abs() > prune_threshold * (1.0 - strength)
    decayed = decayed * mask.float()

    decayed = F.normalize(decayed, p=2, dim=0)
    return decayed, strength


def pure_time_decay(
    feature: torch.Tensor,
    t: float,
    base_lambda: float = 0.08,
    noise_scale: float = 0.03,
    prune_threshold: float = 0.01
):
    """
    Baseline: forget only by time, not identity alignment.
    """
    strength = math.exp(-base_lambda * t)
    decayed = feature * strength
    noise = torch.randn_like(decayed) * noise_scale * (1.0 - strength)
    decayed = decayed + noise
    mask = decayed.abs() > prune_threshold * (1.0 - strength)
    decayed = decayed * mask.float()
    decayed = F.normalize(decayed, p=2, dim=0)
    return decayed, strength


def random_pruning_decay(feature: torch.Tensor, t: float, max_prune: float = 0.75):
    """
    Baseline: random feature pruning increases with time.
    """
    prune_prob = min(max_prune, t / 100.0)
    mask = torch.rand_like(feature) > prune_prob
    decayed = feature * mask.float()
    decayed = F.normalize(decayed, p=2, dim=0)
    strength = 1.0 - prune_prob
    return decayed, strength


def no_forgetting(feature: torch.Tensor, t: float):
    """
    Baseline: memory never changes.
    """
    return feature.clone(), 1.0


# ============================================================
# 8. Build Episodic Linked-List Memory
# ============================================================
# We simulate an agent seeing a sequence of events.
# Some events are successful, some fail.
# Success/failure is simulated here, but in real agent systems it would come
# from reward, task completion, or evaluator feedback.


def build_memory(dataset, num_events=300):
    memory = LinkedListMemory()
    indices = random.sample(range(len(dataset)), num_events)

    for event_id, idx in enumerate(tqdm(indices, desc="Building linked-list memory")):
        image, label = dataset[idx]
        feature = encode_single(image)

        node = MemoryNode(
            event_id=event_id,
            key_feature=feature,
            label=label,
            timestamp=event_id
        )

        node.alignment_score = cosine_alignment(feature, self_facts)

        # Simulated outcome rule:
        # Identity-aligned classes are more likely to be successful.
        if label in IDENTITY_CLASS_IDS:
            success_prob = 0.8
        else:
            success_prob = 0.35

        if random.random() < success_prob:
            node.success_branch.add(feature, label, event_id)
        else:
            node.failure_branch.add(feature, label, event_id)

        memory.append(node)

    return memory


memory = build_memory(train_dataset, num_events=300)
print("Memory size:", len(memory))


# ============================================================
# 9. Apply Forgetting to Memory
# ============================================================

def apply_forgetting(memory, current_time, forgetting_mode="identity", base_lambda=0.08):
    """
    Returns a list of decayed node representations.
    """
    decayed_records = []

    for node in memory.iter_nodes():
        age = current_time - node.timestamp

        if forgetting_mode == "identity":
            decayed_feature, strength = identity_aligned_decay(
                node.key_feature,
                t=age,
                alignment_score=node.alignment_score,
                base_lambda=base_lambda
            )
        elif forgetting_mode == "time":
            decayed_feature, strength = pure_time_decay(
                node.key_feature,
                t=age,
                base_lambda=base_lambda
            )
        elif forgetting_mode == "random":
            decayed_feature, strength = random_pruning_decay(
                node.key_feature,
                t=age
            )
        elif forgetting_mode == "none":
            decayed_feature, strength = no_forgetting(node.key_feature, t=age)
        else:
            raise ValueError(f"Unknown forgetting mode: {forgetting_mode}")

        decayed_records.append({
            "event_id": node.event_id,
            "label": node.label,
            "class_name": class_names[node.label],
            "timestamp": node.timestamp,
            "age": age,
            "alignment_score": node.alignment_score,
            "strength": strength,
            "feature": decayed_feature,
            "is_identity_class": node.label in IDENTITY_CLASS_IDS,
            "success_count": len(node.success_branch.features),
            "failure_count": len(node.failure_branch.features)
        })

    return decayed_records


# ============================================================
# 10. Retrieval Evaluation
# ============================================================
# Given a query image, retrieve nearest memory node.
# We test whether forgetting preserves retrieval for identity-aligned memories.


def retrieve_nearest(query_feature, memory_records, top_k=5):
    features = torch.stack([r["feature"] for r in memory_records])
    query = F.normalize(query_feature, p=2, dim=0).unsqueeze(0)
    features = F.normalize(features, p=2, dim=1)

    sims = torch.matmul(features, query.squeeze(0))
    top_values, top_indices = torch.topk(sims, k=top_k)

    results = []
    for score, idx in zip(top_values, top_indices):
        record = memory_records[idx.item()]
        results.append({
            "score": score.item(),
            "event_id": record["event_id"],
            "label": record["label"],
            "class_name": record["class_name"],
            "is_identity_class": record["is_identity_class"],
            "alignment_score": record["alignment_score"],
            "strength": record["strength"]
        })

    return results


@torch.no_grad()
def evaluate_retrieval(dataset, memory_records, num_queries=150):
    indices = random.sample(range(len(dataset)), num_queries)

    top1_class_correct = 0
    top5_class_correct = 0
    identity_queries = 0
    identity_top1_correct = 0
    identity_top5_correct = 0

    for idx in indices:
        image, label = dataset[idx]
        query_feature = encode_single(image)
        results = retrieve_nearest(query_feature, memory_records, top_k=5)

        top1_label = results[0]["label"]
        top5_labels = [r["label"] for r in results]

        if top1_label == label:
            top1_class_correct += 1
        if label in top5_labels:
            top5_class_correct += 1

        if label in IDENTITY_CLASS_IDS:
            identity_queries += 1
            if top1_label == label:
                identity_top1_correct += 1
            if label in top5_labels:
                identity_top5_correct += 1

    return {
        "top1_acc": top1_class_correct / num_queries,
        "top5_acc": top5_class_correct / num_queries,
        "identity_top1_acc": identity_top1_correct / max(1, identity_queries),
        "identity_top5_acc": identity_top5_correct / max(1, identity_queries),
        "identity_queries": identity_queries
    }


# ============================================================
# 11. Experiment 1: Forgetting Curves
# ============================================================

def experiment_forgetting_curves(memory, times=[0, 10, 25, 50, 75, 100]):
    results = []
    modes = ["none", "time", "random", "identity"]

    for mode in modes:
        for current_time in times:
            records = apply_forgetting(memory, current_time=current_time, forgetting_mode=mode)

            identity_strengths = [r["strength"] for r in records if r["is_identity_class"]]
            non_identity_strengths = [r["strength"] for r in records if not r["is_identity_class"]]

            results.append({
                "mode": mode,
                "time": current_time,
                "identity_strength_mean": float(np.mean(identity_strengths)),
                "non_identity_strength_mean": float(np.mean(non_identity_strengths)),
                "all_strength_mean": float(np.mean([r["strength"] for r in records]))
            })

    return results


curve_results = experiment_forgetting_curves(memory)


def plot_forgetting_curves(curve_results):
    modes = sorted(set(r["mode"] for r in curve_results))

    for mode in modes:
        mode_records = [r for r in curve_results if r["mode"] == mode]
        times = [r["time"] for r in mode_records]
        identity_strength = [r["identity_strength_mean"] for r in mode_records]
        non_identity_strength = [r["non_identity_strength_mean"] for r in mode_records]

        plt.figure(figsize=(7, 5))
        plt.plot(times, identity_strength, marker="o", label="Identity-aligned memories")
        plt.plot(times, non_identity_strength, marker="o", label="Non-identity memories")
        plt.xlabel("Time")
        plt.ylabel("Mean Memory Strength")
        plt.title(f"Forgetting Curve: {mode}")
        plt.legend()
        plt.grid(True)
        plt.show()


plot_forgetting_curves(curve_results)


# ============================================================
# 12. Experiment 2: Retrieval Under Forgetting
# ============================================================

def experiment_retrieval_over_time(memory, test_dataset, times=[0, 10, 25, 50, 75, 100], num_queries=150):
    modes = ["none", "time", "random", "identity"]
    rows = []

    for mode in modes:
        for current_time in tqdm(times, desc=f"Retrieval experiment: {mode}"):
            records = apply_forgetting(memory, current_time=current_time, forgetting_mode=mode)
            metrics = evaluate_retrieval(test_dataset, records, num_queries=num_queries)
            metrics["mode"] = mode
            metrics["time"] = current_time
            rows.append(metrics)

    return rows


retrieval_results = experiment_retrieval_over_time(memory, test_dataset)


def plot_metric(results, metric_name):
    modes = sorted(set(r["mode"] for r in results))

    plt.figure(figsize=(8, 5))
    for mode in modes:
        mode_records = [r for r in results if r["mode"] == mode]
        times = [r["time"] for r in mode_records]
        values = [r[metric_name] for r in mode_records]
        plt.plot(times, values, marker="o", label=mode)

    plt.xlabel("Time")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_metric(retrieval_results, "top1_acc")
plot_metric(retrieval_results, "top5_acc")
plot_metric(retrieval_results, "identity_top1_acc")
plot_metric(retrieval_results, "identity_top5_acc")

print("Retrieval results:")
for row in retrieval_results:
    print(row)


# ============================================================
# 13. Experiment 3: Memory Selectivity Test
# ============================================================
# Paper claim:
# Identity-aligned forgetting should retain aligned memories more strongly
# than non-aligned memories.


def experiment_memory_selectivity(memory, current_time=100):
    modes = ["time", "random", "identity"]
    rows = []

    for mode in modes:
        records = apply_forgetting(memory, current_time=current_time, forgetting_mode=mode)
        identity_strengths = np.array([r["strength"] for r in records if r["is_identity_class"]])
        non_identity_strengths = np.array([r["strength"] for r in records if not r["is_identity_class"]])

        selectivity_gap = identity_strengths.mean() - non_identity_strengths.mean()

        rows.append({
            "mode": mode,
            "identity_mean_strength": identity_strengths.mean(),
            "non_identity_mean_strength": non_identity_strengths.mean(),
            "selectivity_gap": selectivity_gap
        })

    return rows


selectivity_results = experiment_memory_selectivity(memory, current_time=100)
print("\nMemory Selectivity Results:")
for row in selectivity_results:
    print(row)


def plot_selectivity(selectivity_results):
    modes = [r["mode"] for r in selectivity_results]
    gaps = [r["selectivity_gap"] for r in selectivity_results]

    plt.figure(figsize=(7, 5))
    plt.bar(modes, gaps)
    plt.xlabel("Forgetting Mode")
    plt.ylabel("Identity Retention Gap")
    plt.title("Identity Selectivity of Forgetting")
    plt.grid(True, axis="y")
    plt.show()


plot_selectivity(selectivity_results)


# ============================================================
# 14. Experiment 4: Ablation on Lambda
# ============================================================
# Tests how aggressive forgetting changes retrieval.


def experiment_lambda_ablation(memory, test_dataset, lambdas=[0.01, 0.03, 0.05, 0.08, 0.12], current_time=100):
    rows = []

    for lam in lambdas:
        records = apply_forgetting(
            memory,
            current_time=current_time,
            forgetting_mode="identity",
            base_lambda=lam
        )
        metrics = evaluate_retrieval(test_dataset, records, num_queries=150)
        metrics["lambda"] = lam
        rows.append(metrics)

    return rows


lambda_results = experiment_lambda_ablation(memory, test_dataset)
print("\nLambda Ablation Results:")
for row in lambda_results:
    print(row)


def plot_lambda_ablation(lambda_results):
    lambdas = [r["lambda"] for r in lambda_results]
    top1 = [r["top1_acc"] for r in lambda_results]
    identity_top1 = [r["identity_top1_acc"] for r in lambda_results]

    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, top1, marker="o", label="Overall Top-1")
    plt.plot(lambdas, identity_top1, marker="o", label="Identity Top-1")
    plt.xlabel("Base Lambda")
    plt.ylabel("Retrieval Accuracy")
    plt.title("Lambda Ablation: Forgetting Aggressiveness")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_lambda_ablation(lambda_results)


# ============================================================
# 15. Optional: Inspect a Query Retrieval
# ============================================================

def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img.cpu() * std + mean
    return img.clamp(0, 1)


def show_query_retrieval(dataset, memory, query_idx=0, current_time=100, mode="identity"):
    image, label = dataset[query_idx]
    query_feature = encode_single(image)
    records = apply_forgetting(memory, current_time=current_time, forgetting_mode=mode)
    results = retrieve_nearest(query_feature, records, top_k=5)

    plt.figure(figsize=(4, 4))
    plt.imshow(denormalize(image).permute(1, 2, 0))
    plt.title(f"Query: {class_names[label]}")
    plt.axis("off")
    plt.show()

    print("Top retrieved memory nodes:")
    for r in results:
        print(r)


show_query_retrieval(test_dataset, memory, query_idx=10, current_time=100, mode="identity")


# ============================================================
# 16. Paper Claims
# ============================================================
# Results Report:
# 1. Identity-aligned forgetting retains self-relevant memories longer.
# 2. Compared to pure time decay, identity-aligned forgetting preserves
#    retrieval accuracy for self-relevant classes.
# 3. Memory selectivity can be measured as:
#       mean_strength(identity memories) - mean_strength(non-identity memories)
# 4. Lambda controls the aggressiveness-stability tradeoff.
# 5. The linked-list structure preserves event chronology, while the branches
#    represent success/failure outcome histories.
# ============================================================
