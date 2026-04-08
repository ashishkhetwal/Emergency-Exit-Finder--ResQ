
import copy
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from graph import Graph
from pathfinding import PathFinder
from utils import parse_input_file, build_default_graph
from mst import kruskal_mst
from sorting import merge_sort

INT_MAX = math.inf

st.set_page_config(
    page_title="Emergency Exit Finder",
    page_icon="🚨",
    layout="wide",
)

st.markdown("""
<style>
  body { background-color: #0f0f0f; }
  .main { background-color: #111; }
  h1, h2, h3 { color: #f0f0f0; }

  .stAlert > div { border-radius: 10px; }

  .metric-card {
    background: #1e1e2e;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 18px 24px;
    text-align: center;
    margin-bottom: 10px;
  }
  .metric-card .label { font-size: 13px; color: #888; margin-bottom: 4px; }
  .metric-card .value { font-size: 28px; font-weight: 700; color: #7dd3fc; }
  .metric-card .value.danger { color: #f87171; }
  .metric-card .value.safe  { color: #4ade80; }

  .step-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 14px; margin: 4px 0;
    border-radius: 8px; background: #1a1a2e;
    border-left: 3px solid #7dd3fc;
    color: #e2e8f0; font-size: 15px;
  }
  .step-row.emergency { border-left-color: #f87171; }
  .step-num {
    background: #7dd3fc; color: #000;
    border-radius: 50%; width: 24px; height: 24px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 13px; flex-shrink: 0;
  }
  .step-num.emergency { background: #f87171; }

  .path-banner {
    border-radius: 10px; padding: 14px 20px;
    font-size: 16px; font-weight: 600;
    margin-bottom: 16px;
  }
  .path-banner.normal   { background: #1e3a5f; border: 1px solid #3b82f6; color: #bfdbfe; }
  .path-banner.emergency{ background: #3b1f1f; border: 1px solid #ef4444; color: #fecaca; }
  .path-banner.blocked  { background: #2d1f0a; border: 1px solid #f59e0b; color: #fde68a; }

  .sidebar .sidebar-content { background: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

if "graph" not in st.session_state:
    g, src, dst, ow = build_default_graph()
    st.session_state.graph           = g
    st.session_state.source          = src
    st.session_state.destinations    = dst
    st.session_state.original_weights = ow
    st.session_state.blocked_edges   = []   # list of (u,v) currently blocked
    st.session_state.fire_nodes      = []   
    
def graph_to_nx(graph, blocked, path_edges, emg_edges):
    G = nx.Graph()
    n = graph.get_number_of_nodes()
    for i in range(n):
        G.add_node(i, label=graph.get_node_name(i))

    for i in range(n):
        for j in range(i + 1, n):
            w = graph.adjacency_matrix[i][j]
            if w != 0:
                if w != INT_MAX:
                    weight_val = w
                else:
                    weight_val = "🔥"
                G.add_edge(i, j, weight=weight_val)
    return G


def draw_graph(graph, normal_result, emergency_result, blocked_edges, fire_nodes):
    n = graph.get_number_of_nodes()
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    original_weights = st.session_state.original_weights
    all_edges_raw = []
    for i in range(n):
        for j in range(i + 1, n):
            w = graph.adjacency_matrix[i][j]
            orig = original_weights.get((i, j), 0)
            if orig != 0:
                if w == INT_MAX:
                    is_blk = True
                else:
                    is_blk = False
                all_edges_raw.append((i, j, orig, is_blk))

    for item in all_edges_raw:
        u = item[0]
        v = item[1]
        w = item[2]
        is_blk = item[3]
        G.add_edge(u, v, weight=w, blocked=is_blk)

    pos = {}
    floor_nodes = {}
    for i in range(n):
        f = graph.get_node_floor(i)
        if f not in floor_nodes: 
            floor_nodes[f] = []
        floor_nodes[f].append(i)

    for floor_level, nodes in floor_nodes.items():
        x_spacing = 2.0 / (len(nodes) + 1)
        idx = 0
        for node in nodes:
            pos[node] = ((idx + 1) * x_spacing - 1.0, float(floor_level) * 1.5)
            idx += 1

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    # Categorise edges
    norm_path_edges = set()
    if normal_result and normal_result.found():
        for k in range(len(normal_result.path) - 1):
            a = normal_result.path[k]
            b = normal_result.path[k + 1]
            if a < b:
                edge = (a, b)
            else:
                edge = (b, a)
            norm_path_edges.add(edge)

    emg_path_edges = set()
    if emergency_result and emergency_result.found():
        for k in range(len(emergency_result.path) - 1):
            a = emergency_result.path[k]
            b = emergency_result.path[k + 1]
            if a < b:
                edge = (a, b)
            else:
                edge = (b, a)
            emg_path_edges.add(edge)

    blk_set = set()
    for item in blocked_edges:
        u = item[0]
        v = item[1]
        if u < v:
            edge = (u, v)
        else:
            edge = (v, u)
        blk_set.add(edge)

    for edge_tuple in G.edges(data=True):
        u = edge_tuple[0]
        v = edge_tuple[1]
        data = edge_tuple[2]
        
        if u < v:
            key = (u, v)
        else:
            key = (v, u)
            
        if key in blk_set:
            color = "#ef4444"
            width = 3.0
            style = "dashed"
            alpha = 0.8
        elif key in emg_path_edges:
            color = "#249F24"
            width = 3.5
            style = "solid"
            alpha = 1.0
        elif key in norm_path_edges:
            color = "#3b82f6"
            width = 3.5
            style = "solid"
            alpha = 1.0
        else:
            color = "#374151"
            width = 1.5
            style = "solid"
            alpha = 0.6

        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               edge_color=color, width=width,
                               style=style, alpha=alpha, ax=ax)

    # Edge weight labels
    edge_labels = {}
    for edge_tuple in G.edges(data=True):
        u = edge_tuple[0]
        v = edge_tuple[1]
        data = edge_tuple[2]
        
        if u < v:
            key = (u, v)
        else:
            key = (v, u)
            
        if key in blk_set:
            edge_labels[(u, v)] = "🔥"
        else:
            edge_labels[(u, v)] = str(data["weight"])

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_color="#94a3b8", font_size=9, ax=ax,
                                 bbox=dict(boxstyle="round,pad=0.2", fc="#0d0d1a", alpha=0.7, ec="none"))

    src = st.session_state.source
    destinations = st.session_state.destinations
    node_colors = []
    node_sizes  = []
    for i in range(n):
        if i == src:
            node_colors.append("#22c55e")
            node_sizes.append(900)
        elif i in destinations:
            node_colors.append("#a855f7")
            node_sizes.append(900)
        elif i in fire_nodes:
            node_colors.append("#ef4444")
            node_sizes.append(800)
        else:
            node_colors.append("#1e40af")
            node_sizes.append(700)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, ax=ax, alpha=0.95)

    labels = {}
    for i in range(n):
        labels[i] = graph.get_node_name(i)
        
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_color="white", font_size=8,
                            font_weight="bold", ax=ax)


    legend_items = [
        mpatches.Patch(color="#22c55e",  label="Source"),
        mpatches.Patch(color="#a855f7",  label="Destination / EXIT"),
        mpatches.Patch(color="#1e40af",  label="Room / Hallway"),
        mpatches.Patch(color="#ef4444",  label="🔥 Fire / Blocked"),
        mpatches.Patch(color="#3b82f6",  label="Normal Path"),
        mpatches.Patch(color="#f97316",  label="Emergency Re-route"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              facecolor="#1e1e2e", edgecolor="#374151",
              labelcolor="white", fontsize=8)

    ax.axis("off")
    plt.tight_layout()
    return fig


def render_path(result, graph, label, is_emergency):
    if is_emergency:
        css_class = "emergency"
    else:
        css_class = "normal"
        
    if not result.found():
        st.markdown(f'<div class="path-banner blocked">⚠️ {label} — No path found to any exit! Building may be fully blocked.</div>',
                    unsafe_allow_html=True)
        return

    exit_name = graph.get_node_name(result.destination_node)
    
    path_names = []
    for n in result.path:
        path_names.append(graph.get_node_name(n))
    arrow_path = " → ".join(path_names)
    
    if is_emergency:
        icon = "🔥"
    else:
        icon = "🗺️"
        
    st.markdown(f'<div class="path-banner {css_class}">{icon} {label} (Target: {exit_name}): {arrow_path}</div>',
                unsafe_allow_html=True)

    st.markdown(f"**Total distance: `{result.distance}` units**")
    st.markdown("**Step-by-step directions:**")
    for i in range(len(result.path) - 1):
        u = result.path[i]
        v = result.path[i + 1]
        
        if u < v:
            key = (u, v)
        else:
            key = (v, u)
            
        wt = st.session_state.original_weights.get(key, "?")
        
        if is_emergency:
            num_class = "emergency"
        else:
            num_class = ""
            
        st.markdown(
            f'<div class="step-row {num_class}">'
            f'<span class="step-num {num_class}">{i+1}</span>'
            f'<span>{graph.get_node_name(u)} ──({wt})──▶ {graph.get_node_name(v)}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

def render_mst_graph(graph, mst_edges, mst_cost):
    n = graph.get_number_of_nodes()
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
        
    pos = {}
    floor_nodes = {}
    for i in range(n):
        f = graph.get_node_floor(i)
        if f not in floor_nodes: 
            floor_nodes[f] = []
        floor_nodes[f].append(i)

    for floor_level, nodes in floor_nodes.items():
        x_spacing = 2.0 / (len(nodes) + 1)
        idx = 0
        for node in nodes:
            pos[node] = ((idx + 1) * x_spacing - 1.0, float(floor_level) * 1.5)
            idx += 1
            
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    
    mst_set = set()
    for item in mst_edges:
        u = item[0]
        v = item[1]
        if u < v:
            edge = (u, v)
        else:
            edge = (v, u)
        mst_set.add(edge)
    
    # Draw all edges faint, MST edges bold
    all_edges_raw = graph.get_all_edges()
    for item in all_edges_raw:
        u = item[0]
        v = item[1]
        w = item[2]
        if graph.is_blocked(u, v):
            continue
        G.add_edge(u, v, weight=w)
        
    for edge_tuple in G.edges(data=True):
        u = edge_tuple[0]
        v = edge_tuple[1]
        data = edge_tuple[2]
        
        if u < v:
            key = (u, v)
        else:
            key = (v, u)
            
        if key in mst_set:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color="#fbbf24", width=3.5, ax=ax)
            # Label MST edge
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): str(data["weight"])},
                                 font_color="#fbbf24", font_size=10, ax=ax,
                                 bbox=dict(boxstyle="round,pad=0.2", fc="#0d0d1a", alpha=0.9, ec="none"))
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color="#374151", width=1.0, alpha=0.4, style="dashed", ax=ax)
            
    # Nodes
    node_colors = []
    for i in range(n):
        if i in st.session_state.destinations:
            node_colors.append("#a855f7") 
        else:
            node_colors.append("#1e40af")
            
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, ax=ax, alpha=0.95)
    
    labels = {}
    for i in range(n):
        labels[i] = graph.get_node_name(i)
        
    nx.draw_networkx_labels(G, pos, labels=labels, font_color="white", font_size=8, font_weight="bold", ax=ax)
    
    legend_items = [
        mpatches.Patch(color="#fbbf24",  label="MST Alarm Wiring"),
        mpatches.Patch(color="#a855f7",  label="Exit Node"),
        mpatches.Patch(color="#1e40af",  label="Standard Room")
    ]
    ax.legend(handles=legend_items, loc="lower left", facecolor="#1e1e2e", edgecolor="#374151", labelcolor="white", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    return fig


with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    st.markdown("### 📂 Upload Building Map")
    uploaded = st.file_uploader("Upload input.txt", type=["txt"])
    if uploaded:
        try:
            content = uploaded.read().decode("utf-8")
            g, src, dst, ow = parse_input_file(content)
            st.session_state.graph            = g
            st.session_state.source           = src
            st.session_state.destinations     = dst
            st.session_state.original_weights = ow
            st.session_state.blocked_edges    = []
            st.session_state.fire_nodes       = []
            st.success("✅ Building map loaded!")
        except Exception as e:
            st.error(f"❌ Failed to parse file: {e}")

    st.divider()

    # Reset button
    if st.button("🔄 Reset to Default Building", use_container_width=True):
        g, src, dst, ow = build_default_graph()
        st.session_state.graph            = g
        st.session_state.source           = src
        st.session_state.destinations     = dst
        st.session_state.original_weights = ow
        st.session_state.blocked_edges    = []
        st.session_state.fire_nodes       = []
        st.rerun()

    st.divider()

    # Building info
    graph = st.session_state.graph
    n = graph.get_number_of_nodes()
    st.markdown("### 🏢 Building Info")
    st.markdown(f"- **Nodes:** {n}")
    st.markdown(f"- **Edges:** {len(graph.get_all_edges())}")
    
    exit_names_list = []
    for d in st.session_state.destinations:
        exit_names_list.append(graph.get_node_name(d))
    exit_names = ', '.join(exit_names_list)
    
    st.markdown(f"- **Exits:** {exit_names}")

    st.divider()
    st.markdown("### 📍 Current Location")
    
    source_options = {}
    for i in range(n):
        node_name = graph.get_node_name(i)
        source_options[node_name] = i
        
    current_source_name = graph.get_node_name(st.session_state.source)
    try:
        source_idx = list(source_options.keys()).index(current_source_name)
    except ValueError:
        source_idx = 0
        
    selected_source_name = st.selectbox(
        "Select Starting Point", 
        options=list(source_options.keys()), 
        index=source_idx
    )
    
    if source_options[selected_source_name] != st.session_state.source:
        st.session_state.source = source_options[selected_source_name]
        st.rerun()

    st.divider()
    st.markdown("### 🔥 Active Blockages")
    if len(st.session_state.blocked_edges) > 0:
        for item in st.session_state.blocked_edges:
            u = item[0]
            v = item[1]
            st.markdown(f"- `{graph.get_node_name(u)}` ↔ `{graph.get_node_name(v)}`")
    else:
        st.markdown("_No corridors blocked_")

    if len(st.session_state.blocked_edges) > 0:
        if st.button("🧹 Clear All Blockages", use_container_width=True):
            for item in st.session_state.blocked_edges:
                u = item[0]
                v = item[1]
                if u < v:
                    key = (u, v)
                else:
                    key = (v, u)
                wt = st.session_state.original_weights.get(key, 1)
                
            st.session_state.blocked_edges = []
            st.session_state.fire_nodes    = []
            st.rerun()

    st.divider()
    st.markdown("### 🏆 Distance Leaderboard")
    st.caption("Ordered via Merge Sort")
    
    if len(st.session_state.destinations) > 0:
        primary_exit = st.session_state.destinations[0]
    else:
        primary_exit = 0
        
    # Use normal_graph to measure layout distance, or graph for current distance
    distances = PathFinder.get_all_distances(graph, primary_exit)
    
    room_list = []
    for i in range(n):
        room_list.append({"name": graph.get_node_name(i), "dist": distances[i]})
        
    def get_dist(x):
        return x["dist"]
        
    sorted_rooms = merge_sort(room_list, key=get_dist)
    
    for r in sorted_rooms:
        if r["dist"] == 0:
            st.markdown(f"- **{r['name']}**: _Exit_")
        elif r["dist"] == INT_MAX:
            st.markdown(f"- **{r['name']}**: _Unreachable_")
        else:
            st.markdown(f"- **{r['name']}**: `{r['dist']}` pts")


st.markdown("# 🚨 Emergency Exit Finder")
st.markdown("_Graph-based shortest path with real-time corridor blocking — powered by Dijkstra's algorithm_")
st.divider()

graph = st.session_state.graph
n     = graph.get_number_of_nodes()

names = []
for i in range(n):
    names.append(graph.get_node_name(i))

# ---- NORMAL PATH (always unblocked graph) ----
normal_graph = copy.deepcopy(graph)

# Remove all blockages for normal scenario
for item in st.session_state.blocked_edges:
    u = item[0]
    v = item[1]
    if u < v:
        key = (u, v)
    else:
        key = (v, u)
    wt = st.session_state.original_weights.get(key, 1)
    normal_graph.unblock_corridor(u, v, wt)

normal_result = PathFinder.find_shortest_path(
    normal_graph,
    st.session_state.source,
    st.session_state.destinations
)

# ---- EMERGENCY PATH (with blockages applied) ----
emergency_result = None
if len(st.session_state.blocked_edges) > 0:
    emergency_graph = copy.deepcopy(graph)

    emergency_result = PathFinder.find_shortest_path(
        emergency_graph,
        st.session_state.source,
        st.session_state.destinations
    )

col1, col2, col3, col4 = st.columns(4)
with col1:
    if normal_result.found():
        val_cls = "safe"
        nd = normal_result.distance
    else:
        val_cls = "danger"
        nd = "N/A"
        
    st.markdown(f'<div class="metric-card"><div class="label">Normal Distance</div>'
                f'<div class="value {val_cls}">{nd}</div></div>', unsafe_allow_html=True)
                
with col2:
    if emergency_result:
        if emergency_result.found():
            val_cls = "safe"
            ed = emergency_result.distance
        else:
            val_cls = "danger"
            ed = "N/A"
    else:
        val_cls = "safe"
        ed = "—"
        
    st.markdown(f'<div class="metric-card"><div class="label">Emergency Distance</div>'
                f'<div class="value {val_cls}">{ed}</div></div>', unsafe_allow_html=True)
                
with col3:
    blk = len(st.session_state.blocked_edges)
    if blk > 0:
        blk_cls = "danger"
    else:
        blk_cls = "safe"
        
    st.markdown(f'<div class="metric-card"><div class="label">Blocked Corridors</div>'
                f'<div class="value {blk_cls}">{blk}</div></div>', unsafe_allow_html=True)
                
with col4:
    if normal_result.found():
        hops = len(normal_result.path) - 1
    else:
        hops = 0
        
    st.markdown(f'<div class="metric-card"><div class="label">Normal Path Hops</div>'
                f'<div class="value">{hops}</div></div>', unsafe_allow_html=True)

st.divider()

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### 🗺️ Building Graph")
    fig = draw_graph(
        graph,
        normal_result,
        emergency_result,
        st.session_state.blocked_edges,
        st.session_state.fire_nodes,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    #block corridor
    st.markdown("### 🔥 Simulate Fire / Block Corridor")

    edges = graph.get_all_edges()
    edge_options = {}
    for item in edges:
        u = item[0]
        v = item[1]
        w = item[2]
        option_string = f"{graph.get_node_name(u)} ↔ {graph.get_node_name(v)} (dist {w})"
        edge_options[option_string] = (u, v, w)
        
    if len(edge_options) > 0:
        selected_label = st.selectbox("Select corridor to block", list(edge_options.keys()))
        if st.button("🔥 Block this corridor", use_container_width=True, type="primary"):
            item = edge_options[selected_label]
            u = item[0]
            v = item[1]
            w = item[2]
            
            if u < v:
                key = (u, v)
            else:
                key = (v, u)
                
            blocked_keys = []
            for b in st.session_state.blocked_edges:
                a = b[0]
                b_node = b[1]
                if a < b_node:
                    blocked_keys.append((a, b_node))
                else:
                    blocked_keys.append((b_node, a))
                    
            if key not in blocked_keys:
                graph.block_corridor(u, v)
                st.session_state.blocked_edges.append((u, v))
                
                for node in [u, v]:
                    if node not in st.session_state.fire_nodes:
                        st.session_state.fire_nodes.append(node)
                st.rerun()
            else:
                st.warning("This corridor is already blocked.")
    else:
        st.info("No available edges to block.")

    st.divider()

    
    st.markdown("### 🔢 Block by Node IDs")
    c1, c2 = st.columns(2)
    with c1:
        n1 = st.number_input("Node 1", min_value=0, max_value=n - 1, step=1, key="n1")
    with c2:
        n2 = st.number_input("Node 2", min_value=0, max_value=n - 1, step=1, key="n2")
    st.caption(f"Selected: **{graph.get_node_name(int(n1))}** ↔ **{graph.get_node_name(int(n2))}**")

    if st.button("🔥 Block Corridor by ID", use_container_width=True):
        u = int(n1)
        v = int(n2)
        if u == v:
            st.error("Cannot block a node with itself.")
        elif graph.get_edge_weight(u, v) == 0:
            st.error(f"No corridor exists between {graph.get_node_name(u)} and {graph.get_node_name(v)}.")
        else:
            if u < v:
                key = (u, v)
            else:
                key = (v, u)
                
            blocked_keys = []
            for b in st.session_state.blocked_edges:
                a = b[0]
                b_node = b[1]
                if a < b_node:
                    blocked_keys.append((a, b_node))
                else:
                    blocked_keys.append((b_node, a))
                    
            if key not in blocked_keys:
                graph.block_corridor(u, v)
                st.session_state.blocked_edges.append((u, v))
                for node in [u, v]:
                    if node not in st.session_state.fire_nodes:
                        st.session_state.fire_nodes.append(node)
                st.rerun()
            else:
                st.warning("Already blocked.")

    st.divider()

    #Adjacency Matrix 
    with st.expander("🧮 Adjacency Matrix", expanded=False):
        import pandas as pd
        matrix_data = []
        for i in range(n):
            row = []
            for j in range(n):
                w = graph.adjacency_matrix[i][j]
                if w == INT_MAX:
                    row.append("🔥")
                elif w == 0:
                    row.append("-")
                else:
                    row.append(str(w))
            matrix_data.append(row)
        df = pd.DataFrame(matrix_data, index=names, columns=names)
        st.dataframe(df, use_container_width=True)

# --- Routing and MST Tabs ---
st.divider()

tab1, tab2 = st.tabs(["🗺️ Evacuation Routing (Search Algorithms)", "💡 Emergency Wiring (MST)"])

with tab1:
    st.markdown("## 📍 Routing Results")

    path_col1, path_col2 = st.columns(2)

    with path_col1:
        st.markdown("### 🟢 Normal State (Dijkstra)")
        render_path(normal_result, normal_graph, "Lobby → EXIT", is_emergency=False)

    with path_col2:
        st.markdown("### 🔴 Emergency State (Dijkstra)")
        if len(st.session_state.blocked_edges) > 0:
            render_path(emergency_result, emergency_graph, "Re-routed Lobby → EXIT", is_emergency=True)
            if normal_result.found() and emergency_result.found():
                diff = emergency_result.distance - normal_result.distance
                if diff > 0:
                    st.markdown(f"⚠️ **Detour adds `{diff}` extra distance units** due to blockage.")
                elif diff == 0:
                    st.markdown("✅ **Emergency route has the same distance as normal.**")
        else:
            st.info("Block a corridor above to see the emergency re-routing.")

    st.divider()
    st.markdown("### 📊 Algorithm Comparison")
    st.caption("Comparing Pathfinding algorithms in the **current building state** (blocked or normal).")
    
    if len(st.session_state.blocked_edges) > 0:
        g_target = emergency_graph
        target_result = emergency_result
    else:
        g_target = normal_graph
        target_result = normal_result
        
    bfs_result = PathFinder.find_path_bfs(g_target, st.session_state.source, st.session_state.destinations)
    dfs_result = PathFinder.find_path_dfs(g_target, st.session_state.source, st.session_state.destinations)
    
    def get_algo_stats(name, result, g):
        if not result.found():
            return {"Algorithm": name, "Status": "No Path", "Weight Cost": "-", "Hops (Unweighted)": "-", "Nodes Explored": result.visited_count}
        
        cost = 0
        for i in range(len(result.path) - 1):
            u = result.path[i]
            v = result.path[i + 1]
            cost += g.get_edge_weight(u, v)
            
        hops = len(result.path) - 1
        
        path_names = []
        for n in result.path:
            path_names.append(g.get_node_name(n))
        path_str = " → ".join(path_names)
        
        return {"Algorithm": name, "Status": "Path Found", "Weight Cost": cost, "Hops (Unweighted)": hops, "Nodes Explored": result.visited_count, "Path": path_str}

    import pandas as pd
    comp_data = []
    comp_data.append(get_algo_stats("Dijkstra (Weighted)", target_result, g_target))
    comp_data.append(get_algo_stats("BFS (Unweighted)", bfs_result, g_target))
    comp_data.append(get_algo_stats("DFS (Unweighted)", dfs_result, g_target))
    
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

with tab2:
    st.markdown("## 💡 Minimum Spanning Tree (Alarm Wiring)")
    st.markdown("Using **Kruskal's Algorithm**, we can calculate the cheapest way to lay fire alarm wiring such that all rooms are connected to the central system without redundant loops.")
    
    if len(st.session_state.blocked_edges) > 0:
        g_target = emergency_graph
    else:
        g_target = normal_graph
        
    mst_edges, mst_cost = kruskal_mst(g_target)
    
    st.success(f"**Optimal Wiring Cost:** `{mst_cost}` units")
    
    fig = render_mst_graph(g_target, mst_edges, mst_cost)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)