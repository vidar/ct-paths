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


def has_url_field(schema: list) -> bool:
    return any(fld.get("uid") == "url" for fld in schema)


def build_ct_map(content_types: list[dict]) -> dict:
    ct_map = {}
    for ct in content_types:
        uid = ct["uid"]
        schema = ct.get("schema", [])
        ct_map[uid] = {
            "title": ct.get("title", uid),
            "schema": schema,
            "has_url": has_url_field(schema),
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


def build_ref_tree(root_uid: str, ct_map: dict, ref_graph: dict, max_depth: int | None = 2) -> TreeNode:
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
            if not is_cycle and (max_depth is None or child.depth < max_depth):
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


def tree_stats(root: TreeNode) -> tuple[int, int, bool, dict[int, int]]:
    max_depth = 0
    path_count = 0
    has_cycle = False
    refs_per_depth: dict[int, int] = {}

    def traverse(node: TreeNode):
        nonlocal max_depth, path_count, has_cycle
        max_depth = max(max_depth, node.depth)
        if node.is_cycle:
            has_cycle = True
        if not node.children:
            path_count += 1
        if node.depth > 0:
            refs_per_depth[node.depth] = refs_per_depth.get(node.depth, 0) + 1
        for child in node.children:
            traverse(child)

    traverse(root)
    return max_depth, path_count, has_cycle, refs_per_depth


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
    max_depth, path_count, has_cycle, refs_per_depth = tree_stats(root)

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
        depth_line = ""
    else:
        cycle_note = "Contains cycles" if has_cycle else "No cycles"
        subtitle = f"Depth: {max_depth} | Paths: {path_count} | {cycle_note}"
        depth_parts = [
            f"L{d}: {refs_per_depth[d]} refs"
            for d in sorted(refs_per_depth)
        ]
        depth_line = "  |  ".join(depth_parts)

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
    if depth_line:
        ax.text(
            root.x,
            0.5,
            depth_line,
            ha="center",
            va="center",
            fontsize=9,
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
# Interactive HTML generation
# ---------------------------------------------------------------------------


def tree_to_dict(node: TreeNode) -> dict:
    """Serialize a TreeNode to a JSON-compatible dict."""
    return {
        "uid": node.ct_uid,
        "title": node.ct_title,
        "depth": node.depth,
        "isCycle": node.is_cycle,
        "edgeLabel": node.edge_label,
        "children": [tree_to_dict(c) for c in node.children],
    }


def get_tree_max_depth(node: TreeNode) -> int:
    """Return the maximum depth in the tree."""
    if not node.children:
        return node.depth
    return max(get_tree_max_depth(c) for c in node.children)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__ — Reference Tree</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0D1117;color:#C9D1D9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;overflow:hidden;height:100vh}

#header{position:fixed;top:0;left:0;right:0;z-index:100;background:#161B22;border-bottom:1px solid #30363D;padding:10px 20px;display:flex;align-items:center;gap:24px}
#header h1{font-size:15px;font-weight:600;color:#58A6FF;white-space:nowrap}

.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.controls label{font-size:13px;color:#8B949E;margin-right:2px}
.controls input[type="number"]{width:48px;background:#0D1117;border:1px solid #30363D;color:#C9D1D9;padding:3px 6px;border-radius:4px;font-size:13px;text-align:center;-moz-appearance:textfield}
.controls input[type="number"]::-webkit-outer-spin-button,
.controls input[type="number"]::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
.controls button{background:#21262D;border:1px solid #30363D;color:#C9D1D9;padding:3px 10px;border-radius:4px;cursor:pointer;font-size:13px;line-height:1.5}
.controls button:hover{background:#30363D;border-color:#8B949E}
.sep{width:1px;height:20px;background:#30363D;margin:0 4px}

#chart{position:fixed;top:44px;left:0;right:0;bottom:0}

#legend{position:fixed;bottom:16px;left:16px;z-index:100;background:#161B22;border:1px solid #30363D;border-radius:8px;padding:10px 14px;font-size:11px}
.legend-item{display:flex;align-items:center;gap:8px;margin:3px 0}
.legend-swatch{width:14px;height:14px;border-radius:3px;border:2px solid;flex-shrink:0}

.node{cursor:default}
.node.expandable{cursor:pointer}
.node.expandable:hover .node-rect{filter:brightness(1.4)}
.link{fill:none;stroke:#8B949E;stroke-width:1.5}
.edge-label{font-size:10px;fill:#8B949E;font-style:italic;pointer-events:none}
.leaf-counter{font-size:13px;color:#7EE787;font-weight:600}
#ignore-tags{display:flex;gap:4px;flex-wrap:wrap}
.ignore-tag{background:#F8514933;border:1px solid #F85149;color:#F85149;padding:2px 8px;border-radius:4px;font-size:12px;cursor:pointer;display:flex;align-items:center;gap:4px}
.ignore-tag:hover{background:#F8514966}
#ignore-select{background:#0D1117;border:1px solid #30363D;color:#C9D1D9;padding:3px 6px;border-radius:4px;font-size:13px}
</style>
</head>
<body>
<div id="header">
  <h1>__TITLE__ — Reference Tree</h1>
  <div class="controls">
    <label>Depth:</label>
    <button id="btn-minus">&#x2212;</button>
    <input type="number" id="depth-input" value="2" min="0" max="__MAX_DEPTH__">
    <button id="btn-plus">+</button>
    <span class="sep"></span>
    <button id="btn-expand">Expand All</button>
    <button id="btn-collapse">Collapse All</button>
    <span class="sep"></span>
    <button id="btn-reset">Reset View</button>
    <span class="sep"></span>
    <span id="leaf-count" class="leaf-counter"></span>
    <span class="sep"></span>
    <label>Ignore:</label>
    <select id="ignore-select"><option value="">+ Add...</option></select>
    <div id="ignore-tags"></div>
  </div>
</div>

<div id="chart"></div>

<div id="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#58A6FF33;border-color:#58A6FF"></div>Root</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#7EE78733;border-color:#7EE787"></div>Intermediate</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#D2A8FF33;border-color:#D2A8FF"></div>Leaf</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#F8514933;border-color:#F85149;border-style:dashed"></div>Cycle</div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const TREE_DATA = __TREE_DATA__;
const FULL_DEPTH = __MAX_DEPTH__;

const COLORS = {root:"#58A6FF",intermediate:"#7EE787",leaf:"#D2A8FF",cycle:"#F85149",text:"#FFFFFF",edge:"#8B949E",bg:"#0D1117"};
const NODE_H = 32;
const CHAR_W = 8;
const NODE_PAD = 24;

/* ---- hierarchy ---- */
let root = d3.hierarchy(TREE_DATA);
let idCtr = 0;
root.each(d => { d.id = idCtr++; });
root.each(d => { d._children = d.children || null; });

const layout = d3.tree().nodeSize([200, 120]);

/* ---- svg + zoom ---- */
const svg = d3.select("#chart").append("svg").attr("width","100%").attr("height","100%");
const g = svg.append("g");
const zoomBehavior = d3.zoom().scaleExtent([0.05, 4]).on("zoom", e => g.attr("transform", e.transform));
svg.call(zoomBehavior);

const linkGroup = g.append("g");
const labelGroup = g.append("g");
const nodeGroup = g.append("g");

/* ---- helpers ---- */
function nw(d) { var t = d.data.isCycle ? d.data.title + " \u21BA" : d.data.title; return Math.max(100, t.length * CHAR_W + NODE_PAD); }
function nc(d) { if(d.depth===0) return COLORS.root; if(d.data.isCycle) return COLORS.cycle; if(d._children && d._children.length) return COLORS.intermediate; return COLORS.leaf; }
function hasHidden(d) { return d._children && d._children.length > 0 && !d.children; }

function visitAll(node, fn) { fn(node); if(node._children) node._children.forEach(c => visitAll(c, fn)); }

function setVisibleDepth(depth) {
  visitAll(root, d => {
    if(d._children && d._children.length > 0)
      d.children = d.depth < depth ? d._children : null;
  });
}

function diag(s, t) {
  var sy = s.y + NODE_H/2, ty = t.y - NODE_H/2, my = (sy+ty)/2;
  return "M"+s.x+","+sy+" C"+s.x+","+my+" "+t.x+","+my+" "+t.x+","+ty;
}

/* ---- leaf counter ---- */
var uniformDepth = true;
function countLeaves() {
  var c = 0;
  (function v(n){ if(!n.children||!n.children.length) c++; else n.children.forEach(v); })(root);
  return c;
}
function updateLeafCounter() {
  var el = document.getElementById("leaf-count");
  if(uniformDepth){ el.textContent = "Leaves: "+countLeaves(); el.style.display = ""; }
  else { el.style.display = "none"; }
}

/* ---- update ---- */
function update(source) {
  var dur = 400;
  layout(root);

  var nodes = root.descendants();
  var links = root.links();

  /* nodes */
  var node = nodeGroup.selectAll("g.node").data(nodes, d => d.id);

  var ne = node.enter().append("g").attr("class","node")
    .attr("transform","translate("+(source.x0||0)+","+(source.y0||0)+")")
    .style("opacity",0)
    .on("click", (ev, d) => {
      if(!d._children || !d._children.length || d.data.isCycle) return;
      d.children = d.children ? null : d._children;
      update(d);
      uniformDepth = false; updateLeafCounter();
    });

  ne.append("rect").attr("class","node-rect").attr("y",-NODE_H/2).attr("height",NODE_H).attr("rx",6).attr("ry",6);
  ne.append("text").attr("class","node-title").attr("text-anchor","middle").attr("dy","0.35em")
    .attr("fill",COLORS.text).attr("font-size","12px").attr("font-weight","bold")
    .attr("font-family","-apple-system,BlinkMacSystemFont,sans-serif").attr("pointer-events","none");
  ne.append("circle").attr("class","badge-circle");
  ne.append("text").attr("class","badge-text").attr("text-anchor","middle").attr("pointer-events","none");
  ne.append("title");

  var nu = ne.merge(node);
  nu.classed("expandable", d => d._children && d._children.length > 0 && !d.data.isCycle);
  nu.transition().duration(dur).attr("transform", d => "translate("+d.x+","+d.y+")").style("opacity",1);

  nu.select(".node-rect").attr("x", d => -nw(d)/2).attr("width", d => nw(d))
    .attr("fill", d => nc(d)+"33").attr("stroke", d => nc(d)).attr("stroke-width",2)
    .attr("stroke-dasharray", d => d.data.isCycle ? "6,3" : "none");
  nu.select(".node-title").text(d => d.data.isCycle ? d.data.title+" \u21BA" : d.data.title);
  nu.select("title").text(d => d.data.title+" ("+d.data.uid+")");

  nu.select(".badge-circle").attr("cy",NODE_H/2+10).attr("r", d => hasHidden(d)?9:0).attr("fill", d => nc(d));
  nu.select(".badge-text").attr("y",NODE_H/2+10).attr("dy","0.35em").attr("font-size","9px").attr("font-weight","bold").attr("fill",COLORS.bg)
    .text(d => hasHidden(d) ? "+"+d._children.length : "");

  node.exit().transition().duration(dur)
    .attr("transform","translate("+source.x+","+source.y+")").style("opacity",0).remove();

  /* links */
  var link = linkGroup.selectAll("path.link").data(links, d => d.target.id);
  var o0 = {x: source.x0||0, y: source.y0||0};
  var le = link.enter().append("path").attr("class","link").attr("d", diag(o0,o0)).style("opacity",0);
  le.merge(link).transition().duration(dur).attr("d", d => diag(d.source, d.target)).style("opacity",1)
    .attr("stroke-dasharray", d => d.target.data.isCycle ? "6,3" : "none");
  link.exit().transition().duration(dur).attr("d", diag({x:source.x,y:source.y},{x:source.x,y:source.y})).style("opacity",0).remove();

  /* edge labels */
  var el = labelGroup.selectAll("text.edge-label").data(links.filter(d => d.target.data.edgeLabel), d => d.target.id);
  var ele = el.enter().append("text").attr("class","edge-label").attr("text-anchor","middle").attr("x",source.x0||0).attr("y",source.y0||0).style("opacity",0);
  ele.merge(el).transition().duration(dur)
    .attr("x", d => (d.source.x+d.target.x)/2).attr("y", d => (d.source.y+d.target.y)/2+4)
    .style("opacity",1).text(d => d.target.data.edgeLabel);
  el.exit().transition().duration(dur).style("opacity",0).remove();

  nodes.forEach(d => { d.x0 = d.x; d.y0 = d.y; });
}

/* ---- controls ---- */
function setDepth(d) {
  d = Math.max(0, Math.min(FULL_DEPTH, d));
  document.getElementById("depth-input").value = d;
  setVisibleDepth(d);
  update(root);
  uniformDepth = true; updateLeafCounter();
}

function resetView() {
  var b = g.node().getBBox();
  var p = svg.node().getBoundingClientRect();
  if(b.width === 0 || b.height === 0) return;
  var s = Math.min(p.width/(b.width+120), p.height/(b.height+120), 1.5);
  var tx = p.width/2 - (b.x + b.width/2)*s;
  var ty = p.height/2 - (b.y + b.height/2)*s;
  svg.transition().duration(500).call(zoomBehavior.transform, d3.zoomIdentity.translate(tx, ty).scale(s));
}

document.getElementById("btn-minus").onclick = () => setDepth(+document.getElementById("depth-input").value - 1);
document.getElementById("btn-plus").onclick = () => setDepth(+document.getElementById("depth-input").value + 1);
document.getElementById("btn-expand").onclick = () => setDepth(FULL_DEPTH);
document.getElementById("btn-collapse").onclick = () => setDepth(0);
document.getElementById("btn-reset").onclick = resetView;
document.getElementById("depth-input").onchange = function(){ setDepth(+this.value); };

/* ---- ignore filter ---- */
var ignoredTypes = new Set();
var allTypes = new Map();
(function ct(n){ if(!allTypes.has(n.uid)) allTypes.set(n.uid, n.title); if(n.children) n.children.forEach(ct); })(TREE_DATA);

function filterTree(node) {
  var kids = (node.children||[]).filter(function(c){ return !ignoredTypes.has(c.uid); }).map(filterTree);
  return {uid:node.uid,title:node.title,depth:node.depth,isCycle:node.isCycle,edgeLabel:node.edgeLabel,children:kids};
}

function rebuildTree() {
  var depth = +document.getElementById("depth-input").value;
  var filtered = filterTree(TREE_DATA);
  root = d3.hierarchy(filtered);
  idCtr = 0;
  root.each(function(d){ d.id = idCtr++; });
  root.each(function(d){ d._children = d.children || null; });
  nodeGroup.selectAll("*").remove();
  linkGroup.selectAll("*").remove();
  labelGroup.selectAll("*").remove();
  setVisibleDepth(depth);
  uniformDepth = true;
  root.x0 = 0; root.y0 = 0;
  update(root);
  updateLeafCounter();
  requestAnimationFrame(function(){ requestAnimationFrame(resetView); });
}

function updateIgnoreTags() {
  var container = document.getElementById("ignore-tags");
  var sel = document.getElementById("ignore-select");
  container.innerHTML = "";
  ignoredTypes.forEach(function(uid) {
    var tag = document.createElement("span");
    tag.className = "ignore-tag";
    tag.textContent = (allTypes.get(uid)||uid) + " \u00d7";
    tag.onclick = function(){ ignoredTypes.delete(uid); updateIgnoreTags(); rebuildTree(); };
    container.appendChild(tag);
  });
  sel.disabled = ignoredTypes.size >= 5;
  Array.from(sel.options).forEach(function(opt){ if(opt.value) opt.disabled = ignoredTypes.has(opt.value); });
}

(function() {
  var sel = document.getElementById("ignore-select");
  var sorted = Array.from(allTypes.entries()).filter(function(e){ return e[0] !== TREE_DATA.uid; }).sort(function(a,b){ return a[1].localeCompare(b[1]); });
  sorted.forEach(function(e) {
    var opt = document.createElement("option");
    opt.value = e[0]; opt.textContent = e[1];
    sel.appendChild(opt);
  });
  sel.onchange = function() {
    if(!this.value || ignoredTypes.size >= 5){ this.value = ""; return; }
    ignoredTypes.add(this.value);
    this.value = "";
    updateIgnoreTags();
    rebuildTree();
  };
})();

/* ---- init ---- */
setVisibleDepth(2);
root.x0 = 0; root.y0 = 0;
update(root);
updateLeafCounter();
requestAnimationFrame(() => requestAnimationFrame(resetView));
</script>
</body>
</html>"""


def generate_html(root: TreeNode, output_dir: str) -> str:
    """Generate an interactive HTML visualization of the reference tree."""
    data_json = json.dumps(tree_to_dict(root), indent=2)
    max_depth = get_tree_max_depth(root)

    html = _HTML_TEMPLATE
    html = html.replace("__TREE_DATA__", data_json)
    html = html.replace("__TITLE__", root.ct_title)
    html = html.replace("__MAX_DEPTH__", str(max_depth))

    output_path = os.path.join(output_dir, f"{root.ct_uid}_reference_tree.html")
    with open(output_path, "w") as f:
        f.write(html)

    return output_path


# ---------------------------------------------------------------------------
# Index generation
# ---------------------------------------------------------------------------


def generate_index(results: list[dict], ct_map: dict, output_dir: str):
    lines = [
        "# Reference Tree Diagrams",
        "",
        f"Generated from {len(ct_map)} content types.",
        "",
        f"## Content Types with URL Field ({len(results)} diagrams)",
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
    parser.add_argument("--output", default="./graphs", help="Output directory")
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

    page_types = ["page_wrapper"]
    print(f"  Types with URL field: {len(page_types)}\n")

    results = []
    for uid in sorted(page_types):
        full_tree = build_ref_tree(uid, ct_map, ref_graph, max_depth=None)
        html_path = generate_html(full_tree, args.output)
        print(f"  Generated: {os.path.basename(html_path)}")

        max_depth, _, has_cycle, _ = tree_stats(full_tree)
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

    generate_index(results, ct_map, args.output)
    print(f"  Generated: _index.md")
    print(f"\nDone. {len(results)} diagrams in {args.output}/")


if __name__ == "__main__":
    main()
