
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from backend import Graph, PathFinder, parse_input_file, build_default_graph

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
    st.session_state.destination     = dst
    st.session_state.original_weights = ow
    st.session_state.blocked_edges   = []   # list of (u,v) currently blocked
    st.session_state.fire_nodes      = []   
    
def graph_to_nx(graph: Graph, blocked: list, path_edges: list, emg_edges: list):
    G = nx.Graph()
    n = graph.get_number_of_nodes()
    for i in range(n):
        G.add_node(i, label=graph.get_node_name(i))

    for i in range(n):
        for j in range(i + 1, n):
            w = graph.adjacency_matrix[i][j]
            if w != 0:
                G.add_edge(i, j, weight=w if w != INT_MAX else "🔥")
    return G


def draw_graph(graph: Graph, normal_result, emergency_result, blocked_edges, fire_nodes):
    n = graph.get_number_of_nodes()
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    original_weights = st.session_state.original_weights
    all_edges_raw = []
    for i in range(n):
        for j in range(i + 1, n):
            w = graph.adjacency_matrix[i][j]
            orig = original_weights.get((min(i, j), max(i, j)), 0)
            if orig != 0:
                all_edges_raw.append((i, j, orig, w == INT_MAX))

    for u, v, w, is_blk in all_edges_raw:
        G.add_edge(u, v, weight=w, blocked=is_blk)

    pos = nx.spring_layout(G, seed=42, k=2.5)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    # Categorise edges
    norm_path_edges = set()
    if normal_result and normal_result.found():
        for k in range(len(normal_result.path) - 1):
            a, b = normal_result.path[k], normal_result.path[k + 1]
            norm_path_edges.add((min(a, b), max(a, b)))

    emg_path_edges = set()
    if emergency_result and emergency_result.found():
        for k in range(len(emergency_result.path) - 1):
            a, b = emergency_result.path[k], emergency_result.path[k + 1]
            emg_path_edges.add((min(a, b), max(a, b)))

    blk_set = {(min(u, v), max(u, v)) for u, v in blocked_edges}

    for u, v, data in G.edges(data=True):
        key = (min(u, v), max(u, v))
        if key in blk_set:
            color, width, style, alpha = "#ef4444", 3.0, "dashed", 0.8
        elif key in emg_path_edges:
            color, width, style, alpha = "#f97316", 3.5, "solid", 1.0
        elif key in norm_path_edges:
            color, width, style, alpha = "#3b82f6", 3.5, "solid", 1.0
        else:
            color, width, style, alpha = "#374151", 1.5, "solid", 0.6

        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               edge_color=color, width=width,
                               style=style, alpha=alpha, ax=ax)

    # Edge weight labels
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        key = (min(u, v), max(u, v))
        if key in blk_set:
            edge_labels[(u, v)] = "🔥"
        else:
            edge_labels[(u, v)] = str(data["weight"])

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_color="#94a3b8", font_size=9, ax=ax,
                                 bbox=dict(boxstyle="round,pad=0.2", fc="#0d0d1a", alpha=0.7, ec="none"))

  
    src = st.session_state.source
    dst = st.session_state.destination
    node_colors = []
    node_sizes  = []
    for i in range(n):
        if i == src:
            node_colors.append("#22c55e"); node_sizes.append(900)
        elif i == dst:
            node_colors.append("#a855f7"); node_sizes.append(900)
        elif i in fire_nodes:
            node_colors.append("#ef4444"); node_sizes.append(800)
        else:
            node_colors.append("#1e40af"); node_sizes.append(700)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, ax=ax, alpha=0.95)

    labels = {i: graph.get_node_name(i) for i in range(n)}
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


def render_path(result, graph: Graph, label: str, is_emergency: bool):
    css_class = "emergency" if is_emergency else "normal"
    if not result.found():
        st.markdown(f'<div class="path-banner blocked">⚠️ {label} — No path found! Building may be fully blocked.</div>',
                    unsafe_allow_html=True)
        return

    arrow_path = " → ".join(graph.get_node_name(n) for n in result.path)
    st.markdown(f'<div class="path-banner {css_class}">{"🔥" if is_emergency else "🗺️"} {label}: {arrow_path}</div>',
                unsafe_allow_html=True)

    st.markdown(f"**Total distance: `{result.distance}` units**")
    st.markdown("**Step-by-step directions:**")
    for i in range(len(result.path) - 1):
        u = result.path[i]
        v = result.path[i + 1]
        wt = st.session_state.original_weights.get((min(u, v), max(u, v)), "?")
        num_class = "emergency" if is_emergency else ""
        st.markdown(
            f'<div class="step-row {num_class}">'
            f'<span class="step-num {num_class}">{i+1}</span>'
            f'<span>{graph.get_node_name(u)} ──({wt})──▶ {graph.get_node_name(v)}</span>'
            f'</div>',
            unsafe_allow_html=True
        )


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
            st.session_state.destination      = dst
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
        st.session_state.destination      = dst
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
    st.markdown(f"- **Source:** {graph.get_node_name(st.session_state.source)}")
    st.markdown(f"- **Exit:** {graph.get_node_name(st.session_state.destination)}")

    st.divider()
    st.markdown("### 🔥 Active Blockages")
    if st.session_state.blocked_edges:
        for (u, v) in st.session_state.blocked_edges:
            st.markdown(f"- `{graph.get_node_name(u)}` ↔ `{graph.get_node_name(v)}`")
    else:
        st.markdown("_No corridors blocked_")

    if st.session_state.blocked_edges:
        if st.button("🧹 Clear All Blockages", use_container_width=True):
            for (u, v) in st.session_state.blocked_edges:
                wt = st.session_state.original_weights.get((min(u, v), max(u, v)), 1)
                st.session_state.graph.unblock_corridor(u, v, wt)
            st.session_state.blocked_edges = []
            st.session_state.fire_nodes    = []
            st.rerun()


st.markdown("# 🚨 Emergency Exit Finder")
st.markdown("_Graph-based shortest path with real-time corridor blocking — powered by Dijkstra's algorithm_")
st.divider()

graph = st.session_state.graph
n     = graph.get_number_of_nodes()
names = [graph.get_node_name(i) for i in range(n)]

#  Normal path 
normal_result = PathFinder.find_shortest_path(
    graph,
    st.session_state.source,
    st.session_state.destination
)

#  Emergency path 
emergency_result = None
if st.session_state.blocked_edges:
    emergency_result = PathFinder.find_shortest_path(
        graph,
        st.session_state.source,
        st.session_state.destination
    )

col1, col2, col3, col4 = st.columns(4)
with col1:
    val_cls = "safe" if normal_result.found() else "danger"
    nd = normal_result.distance if normal_result.found() else "N/A"
    st.markdown(f'<div class="metric-card"><div class="label">Normal Distance</div>'
                f'<div class="value {val_cls}">{nd}</div></div>', unsafe_allow_html=True)
with col2:
    if emergency_result:
        val_cls = "safe" if emergency_result.found() else "danger"
        ed = emergency_result.distance if emergency_result.found() else "N/A"
    else:
        val_cls, ed = "safe", "—"
    st.markdown(f'<div class="metric-card"><div class="label">Emergency Distance</div>'
                f'<div class="value {val_cls}">{ed}</div></div>', unsafe_allow_html=True)
with col3:
    blk = len(st.session_state.blocked_edges)
    blk_cls = "danger" if blk > 0 else "safe"
    st.markdown(f'<div class="metric-card"><div class="label">Blocked Corridors</div>'
                f'<div class="value {blk_cls}">{blk}</div></div>', unsafe_allow_html=True)
with col4:
    hops = len(normal_result.path) - 1 if normal_result.found() else 0
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
    edge_options = {
        f"{graph.get_node_name(u)} ↔ {graph.get_node_name(v)} (dist {w})": (u, v, w)
        for u, v, w in edges
    }
    if edge_options:
        selected_label = st.selectbox("Select corridor to block", list(edge_options.keys()))
        if st.button("🔥 Block this corridor", use_container_width=True, type="primary"):
            u, v, w = edge_options[selected_label]
            key = (min(u, v), max(u, v))
            if key not in [(min(a, b), max(a, b)) for a, b in st.session_state.blocked_edges]:
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
        u, v = int(n1), int(n2)
        if u == v:
            st.error("Cannot block a node with itself.")
        elif graph.get_edge_weight(u, v) == 0:
            st.error(f"No corridor exists between {graph.get_node_name(u)} and {graph.get_node_name(v)}.")
        else:
            key = (min(u, v), max(u, v))
            if key not in [(min(a, b), max(a, b)) for a, b in st.session_state.blocked_edges]:
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

#Path
st.divider()
st.markdown("## 📍 Routing Results")

path_col1, path_col2 = st.columns(2)

with path_col1:
    st.markdown("### 🟢 Normal State")
    render_path(normal_result, graph, "Lobby → EXIT", is_emergency=False)

with path_col2:
    st.markdown("### 🔴 Emergency State")
    if st.session_state.blocked_edges:
        render_path(emergency_result, graph, "Re-routed Lobby → EXIT", is_emergency=True)
        if normal_result.found() and emergency_result.found():
            diff = emergency_result.distance - normal_result.distance
            if diff > 0:
                st.markdown(f"⚠️ **Detour adds `{diff}` extra distance units** due to blockage.")
            elif diff == 0:
                st.markdown("✅ **Emergency route has the same distance as normal.**")
    else:
        st.info("Block a corridor above to see the emergency re-routing.")


st.divider()
st.markdown(
    "<center style='color:#555; font-size:13px;'>Emergency Exit Finder · "
    "C++ backend ported to Python · Dijkstra's Algorithm · "
    "PBL 4th Semester Project</center>",
    unsafe_allow_html=True
)
