#!/usr/bin/env python3
"""Generate reference tree diagrams for Contentstack content types."""

import argparse
import json
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#0D1117",
    "root": "#58A6FF",
    "intermediate": "#7EE787",
    "leaf": "#D2A8FF",
    "cycle": "#F85149",
    "text": "#FFFFFF",
    "edge": "#8B949E",
    "edge_label": "#C9D1D9",
}

H_SPACING = 2.2  # horizontal units per leaf node
V_SPACING = 2.8  # vertical units per depth level
NODE_HEIGHT = 0.6
CHAR_WIDTH = 0.12  # approximate width per character for node sizing
MIN_NODE_WIDTH = 1.6

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RefEdge:
    from_ct: str
    to_ct: str
    field_name: str
    block_title: str  # title of enclosing block, or "" if top-level


@dataclass
class TreeNode:
    ct_uid: str
    ct_title: str
    depth: int
    is_cycle: bool
    edge_label: str
    children: list = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    leaf_count: int = 1


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def load_content_types(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "content_types" in data:
        return data["content_types"]
    if isinstance(data, list):
        return data
    raise ValueError("Expected a JSON object with 'content_types' or a JSON array")


def build_ct_map(content_types: list[dict]) -> dict:
    ct_map = {}
    for ct in content_types:
        uid = ct["uid"]
        ct_map[uid] = {
            "title": ct.get("title", uid),
            "schema": ct.get("schema", []),
            "is_page": ct.get("options", {}).get("is_page", False),
        }
    return ct_map


def find_refs(fields: list, ct_uid: str, path: str = "", block_title: str = "") -> list[RefEdge]:
    refs = []
    for fld in fields:
        fld_path = f"{path}.{fld['uid']}"
        dt = fld.get("data_type", "")

        if dt == "reference" and isinstance(fld.get("reference_to"), list):
            for target in fld["reference_to"]:
                if target != "sys_assets":
                    refs.append(
                        RefEdge(
                            from_ct=ct_uid,
                            to_ct=target,
                            field_name=fld.get("display_name", fld["uid"]),
                            block_title=block_title,
                        )
                    )

        if dt == "group" and "schema" in fld:
            refs.extend(find_refs(fld["schema"], ct_uid, fld_path, block_title))

        if dt == "blocks" and "blocks" in fld:
            for block in fld["blocks"]:
                refs.extend(
                    find_refs(
                        block.get("schema", []),
                        ct_uid,
                        f"{fld_path}.{block['uid']}",
                        block.get("title", block["uid"]),
                    )
                )

        if dt == "global_field" and "schema" in fld:
            refs.extend(find_refs(fld["schema"], ct_uid, fld_path, block_title))

    return refs


def build_ref_graph(content_types: list[dict]) -> dict[str, list[RefEdge]]:
    graph = {}
    for ct in content_types:
        uid = ct["uid"]
        edges = find_refs(ct.get("schema", []), uid)
        if edges:
            graph[uid] = edges
    return graph


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------


def build_ref_tree(root_uid: str, ct_map: dict, ref_graph: dict) -> TreeNode:
    root_title = ct_map.get(root_uid, {}).get("title", root_uid)
    root = TreeNode(ct_uid=root_uid, ct_title=root_title, depth=0, is_cycle=False, edge_label="")

    def expand(node: TreeNode, ancestors: set[str]):
        edges = ref_graph.get(node.ct_uid, [])
        # Group edges by target, collecting labels
        targets: dict[str, list[str]] = {}
        for edge in edges:
            if edge.to_ct not in targets:
                targets[edge.to_ct] = []
            label = edge.block_title if edge.block_title else edge.field_name
            if label not in targets[edge.to_ct]:
                targets[edge.to_ct].append(label)

        for target_uid, labels in targets.items():
            label_str = "via " + ", ".join(labels)
            target_title = ct_map.get(target_uid, {}).get("title", target_uid)
            is_cycle = target_uid in ancestors
            child = TreeNode(
                ct_uid=target_uid,
                ct_title=target_title,
                depth=node.depth + 1,
                is_cycle=is_cycle,
                edge_label=label_str,
            )
            node.children.append(child)
            if not is_cycle:
                expand(child, ancestors | {target_uid})

    expand(root, {root_uid})
    return root


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def compute_leaf_counts(node: TreeNode):
    if not node.children:
        node.leaf_count = 1
        return
    for child in node.children:
        compute_leaf_counts(child)
    node.leaf_count = sum(c.leaf_count for c in node.children)


def assign_positions(node: TreeNode, left_x: float, depth: int):
    node.y = -depth * V_SPACING
    total_width = node.leaf_count * H_SPACING
    node.x = left_x + total_width / 2

    current_x = left_x
    for child in node.children:
        assign_positions(child, current_x, depth + 1)
        current_x += child.leaf_count * H_SPACING


# ---------------------------------------------------------------------------
# Tree stats
# ---------------------------------------------------------------------------


def tree_stats(root: TreeNode) -> tuple[int, int, bool]:
    max_depth = 0
    path_count = 0
    has_cycle = False

    def walk(node: TreeNode):
        nonlocal max_depth, path_count, has_cycle
        max_depth = max(max_depth, node.depth)
        if node.is_cycle:
            has_cycle = True
        if not node.children:
            path_count += 1

    def traverse(node: TreeNode):
        walk(node)
        for child in node.children:
            traverse(child)

    traverse(root)
    return max_depth, path_count, has_cycle


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def node_color(node: TreeNode, is_root: bool) -> str:
    if is_root:
        return COLORS["root"]
    if node.is_cycle:
        return COLORS["cycle"]
    if node.children:
        return COLORS["intermediate"]
    return COLORS["leaf"]


def node_width(title: str) -> float:
    return max(MIN_NODE_WIDTH, len(title) * CHAR_WIDTH + 0.4)


def draw_node(ax, node: TreeNode, is_root: bool = False):
    color = node_color(node, is_root)
    linestyle = "--" if node.is_cycle else "-"
    w = node_width(node.ct_title)
    h = NODE_HEIGHT

    label = node.ct_title
    if node.is_cycle:
        label += " \u21BA"  # cycle arrow symbol

    box = FancyBboxPatch(
        (node.x - w / 2, node.y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.1",
        facecolor=color + "33",
        edgecolor=color,
        linewidth=2,
        linestyle=linestyle,
    )
    ax.add_patch(box)
    ax.text(
        node.x,
        node.y,
        label,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=COLORS["text"],
        fontfamily="sans-serif",
    )


def draw_edge(ax, parent: TreeNode, child: TreeNode):
    ax.annotate(
        "",
        xy=(child.x, child.y + NODE_HEIGHT / 2),
        xytext=(parent.x, parent.y - NODE_HEIGHT / 2),
        arrowprops=dict(
            arrowstyle="-|>",
            color=COLORS["edge"],
            linewidth=1.5,
            connectionstyle="arc3,rad=0",
        ),
    )
    # Edge label at midpoint
    mid_x = (parent.x + child.x) / 2
    mid_y = (parent.y + child.y) / 2
    ax.text(
        mid_x,
        mid_y,
        child.edge_label,
        ha="center",
        va="center",
        fontsize=7,
        color=COLORS["edge_label"],
        fontstyle="italic",
        fontfamily="sans-serif",
        bbox=dict(boxstyle="round,pad=0.15", facecolor=COLORS["bg"], edgecolor="none"),
    )


def render_tree(root: TreeNode, output_dir: str) -> tuple[str, int, int, bool]:
    compute_leaf_counts(root)
    assign_positions(root, 0, 0)
    max_depth, path_count, has_cycle = tree_stats(root)

    # Figure sizing
    width = max(8, root.leaf_count * H_SPACING + 2)
    height = max(4, (max_depth + 1) * V_SPACING + 3)

    fig, ax = plt.subplots(figsize=(width, height))
    fig.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(-1, root.leaf_count * H_SPACING + 1)
    ax.set_ylim(-max_depth * V_SPACING - 1.5, 2.5)
    ax.axis("off")

    # Title and subtitle
    title = f"{root.ct_title} \u2014 Reference Tree"
    if not root.children:
        subtitle = "No outgoing references"
    else:
        cycle_note = "Contains cycles" if has_cycle else "No cycles"
        subtitle = f"Depth: {max_depth} | Paths: {path_count} | {cycle_note}"

    ax.text(
        root.x,
        1.8,
        title,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
        fontfamily="sans-serif",
    )
    ax.text(
        root.x,
        1.1,
        subtitle,
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["edge_label"],
        fontfamily="sans-serif",
    )

    # Draw edges first (behind nodes)
    def draw_edges(node: TreeNode):
        for child in node.children:
            draw_edge(ax, node, child)
            draw_edges(child)

    draw_edges(root)

    # Draw nodes
    def draw_nodes(node: TreeNode, is_root: bool = False):
        draw_node(ax, node, is_root)
        for child in node.children:
            draw_nodes(child)

    draw_nodes(root, is_root=True)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["root"] + "66", edgecolor=COLORS["root"], label="Root"),
        mpatches.Patch(
            facecolor=COLORS["intermediate"] + "66",
            edgecolor=COLORS["intermediate"],
            label="Intermediate",
        ),
        mpatches.Patch(facecolor=COLORS["leaf"] + "66", edgecolor=COLORS["leaf"], label="Leaf"),
        mpatches.Patch(
            facecolor=COLORS["cycle"] + "66",
            edgecolor=COLORS["cycle"],
            label="Cycle",
            linestyle="--",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        facecolor=COLORS["bg"],
        edgecolor=COLORS["edge"],
        labelcolor=COLORS["text"],
        fontsize=8,
    )

    # Save
    output_path = os.path.join(output_dir, f"{root.ct_uid}_reference_tree.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"], edgecolor="none")
    plt.close(fig)

    return output_path, max_depth, path_count, has_cycle


# ---------------------------------------------------------------------------
# Index generation
# ---------------------------------------------------------------------------


def generate_index(results: list[dict], ct_map: dict, output_dir: str):
    lines = [
        "# Reference Tree Diagrams",
        "",
        f"Generated from {len(ct_map)} content types.",
        "",
        f"## Page Types ({len(results)} diagrams)",
        "",
        "| Content Type | UID | Outgoing Refs | Max Depth | Cycles |",
        "|---|---|---|---|---|",
    ]

    for r in sorted(results, key=lambda x: x["title"]):
        cycles = "Yes" if r["has_cycle"] else "No"
        lines.append(
            f"| [{r['title']}]({r['uid']}_reference_tree.png) "
            f"| `{r['uid']}` | {r['ref_count']} | {r['max_depth']} | {cycles} |"
        )

    lines.extend(
        [
            "",
            "## Legend",
            "",
            "- **Blue**: Root node (the content type being diagrammed)",
            "- **Green**: Intermediate node (has further references)",
            "- **Purple**: Leaf node (no outgoing references)",
            "- **Red dashed**: Cycle node (already visited on this path)",
        ]
    )

    with open(os.path.join(output_dir, "_index.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference tree diagrams for Contentstack content types."
    )
    parser.add_argument("--input", required=True, help="Path to content types JSON file")
    parser.add_argument("--output", default="./graphs", help="Output directory for PNGs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    content_types = load_content_types(args.input)
    ct_map = build_ct_map(content_types)
    ref_graph = build_ref_graph(content_types)

    # Print discovered edges for verification
    total_edges = 0
    for uid, edges in sorted(ref_graph.items()):
        targets: dict[str, list[str]] = {}
        for e in edges:
            if e.to_ct not in targets:
                targets[e.to_ct] = []
            label = e.block_title if e.block_title else e.field_name
            if label not in targets[e.to_ct]:
                targets[e.to_ct].append(label)
        for target, labels in targets.items():
            print(f"  {uid:20s} -> {target:20s} (via {', '.join(labels)})")
            total_edges += 1
    print(f"\n  Total unique edges: {total_edges}")

    page_types = [uid for uid, ct in ct_map.items() if ct["is_page"]]
    print(f"  Page types: {len(page_types)}\n")

    results = []
    for uid in sorted(page_types):
        tree = build_ref_tree(uid, ct_map, ref_graph)
        out_path, max_depth, path_count, has_cycle = render_tree(tree, args.output)

        ref_count = len(ref_graph.get(uid, []))
        # Count unique targets for the summary
        unique_targets = len(set(e.to_ct for e in ref_graph.get(uid, [])))
        results.append(
            {
                "uid": uid,
                "title": ct_map[uid]["title"],
                "ref_count": unique_targets,
                "max_depth": max_depth,
                "has_cycle": has_cycle,
            }
        )
        print(f"  Generated: {os.path.basename(out_path)}")

    generate_index(results, ct_map, args.output)
    print(f"  Generated: _index.md")
    print(f"\nDone. {len(results)} diagrams in {args.output}/")


if __name__ == "__main__":
    main()
