import copy
import time
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
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
.metric-card .value.safe { color: #4ade80; }
.metric-card .value.warn { color: #fbbf24; }
.step-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 14px; margin: 4px 0;
    border-radius: 8px; background: #1a1a2e;
    border-left: 3px solid #7dd3fc;
    color: #e2e8f0; font-size: 15px;
}
.step-row.emergency { border-left-color: #f87171; }
.step-row.stair { border-left-color: #fbbf24; }
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
.path-banner.normal  { background: #1e3a5f; border: 1px solid #3b82f6; color: #bfdbfe; }
.path-banner.emergency{ background: #3b1f1f; border: 1px solid #ef4444; color: #fecaca; }
.path-banner.blocked { background: #2d1f0a; border: 1px solid #f59e0b; color: #fde68a; }
.path-banner.tunnel  { background: #1a1a00; border: 1px solid #facc15; color: #fef08a; }
.tunnel-banner {
    background: #1a1a00; border: 2px solid #facc15;
    border-radius: 10px; padding: 12px 18px;
    color: #fef08a; font-size: 15px; font-weight: 700;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "graph" not in st.session_state:
    g, src, dst, ow = build_default_graph()
    st.session_state.graph = g
    st.session_state.source = src
    st.session_state.destinations = dst
    st.session_state.original_weights = ow
    st.session_state.blocked_edges = []
    st.session_state.fire_nodes = []
    st.session_state.animate_path = False
    st.session_state.anim_step = 0


# ─────────────────────────────────────────────
# TUNNEL LOGIC  (adds tunnel edges when no normal path exists)
# ─────────────────────────────────────────────
TUNNEL_WEIGHT = 50   # high enough to be last resort, but realistic enough to display

def add_tunnel_edges(graph, destinations, source):
    """
    Adds a tunnel ONLY from the source node directly to each exit.
    This represents an emergency underground escape from exactly where the person is.
    Only adds tunnel if no existing real edge already connects source to that exit.
    Returns a list of (u, v) tunnel edges added.
    """
    tunnels = []
    for d in destinations:
        # Only add tunnel if there is no existing direct edge between source and exit
        if graph.adjacency_matrix[source][d] == 0:
            graph.adjacency_matrix[source][d] = TUNNEL_WEIGHT
            graph.adjacency_matrix[d][source] = TUNNEL_WEIGHT
            tunnels.append((source, d))
    return tunnels


# ─────────────────────────────────────────────
# DRAW GRAPH
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# IMPROVED 3D INTERACTIVE GRAPH VISUALIZATION
# Powered by Plotly 3D
# ─────────────────────────────────────────────

def draw_graph(graph, normal_result, emergency_result, blocked_edges, fire_nodes,
               tunnel_edges=None, active_anim_result=None):
    n = graph.get_number_of_nodes()
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    original_weights = st.session_state.original_weights

    # Add edges to NetworkX graph for layout structure
    all_edges_raw = []
    for i in range(n):
        for j in range(i + 1, n):
            w = graph.adjacency_matrix[i][j]
            orig = original_weights.get((i, j), 0)
            if orig != 0:
                all_edges_raw.append((i, j, orig, w == INT_MAX))

    for item in all_edges_raw:
        u, v, w, is_blk = item
        G.add_edge(u, v, weight=w, blocked=is_blk)

    pos = {}
    floor_groups = {}
    
    # Group nodes by floor
    for i in range(n):
        floor = graph.get_node_floor(i)
        floor_groups.setdefault(floor, []).append(i)

    for floor, nodes in floor_groups.items():
        subgraph = nx.subgraph(G, nodes)
        if len(nodes) == 1:
            pos[nodes[0]] = (0, 0, floor * 30)
        else:
            sub_pos = nx.spring_layout(subgraph, k=2.0, iterations=50, seed=42 + floor)
            for node in nodes:
                x, y = sub_pos[node]
                pos[node] = (x * 40, y * 40, floor * 30)  

    fig = go.Figure()

    blk_set = set([(min(u,v), max(u,v)) for u,v in blocked_edges])
    tunnel_set = set([(min(u,v), max(u,v)) for u,v in (tunnel_edges or [])])
    
    active_path_edges = set()
    if active_anim_result and active_anim_result.found():
        for k in range(len(active_anim_result.path) - 1):
            a, b = active_anim_result.path[k], active_anim_result.path[k + 1]
            active_path_edges.add((min(a, b), max(a, b)))

    edge_x, edge_y, edge_z = [], [], []
    blk_x, blk_y, blk_z = [], [], []
    tun_x, tun_y, tun_z = [], [], []
    path_x, path_y, path_z = [], [], []

    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        key = (min(u,v), max(u,v))

        if key in blk_set:
            blk_x.extend([x0, x1, None])
            blk_y.extend([y0, y1, None])
            blk_z.extend([z0, z1, None])
        elif key in tunnel_set:
            tun_x.extend([x0, x1, None])
            tun_y.extend([y0, y1, None])
            tun_z.extend([z0, z1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        if key in active_path_edges and key not in blk_set:
            path_x.extend([x0, x1, None])
            path_y.extend([y0, y1, None])
            path_z.extend([z0, z1, None])

    # Standard edges
    fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines',
        line=dict(color='rgba(150,150,150,0.4)', width=2), hoverinfo='none', showlegend=False))
    
    # Active Path
    if path_x:
        fig.add_trace(go.Scatter3d(x=path_x, y=path_y, z=path_z, mode='lines',
            line=dict(color='#22c55e', width=8), name='Evacuation Route'))

    # Blocked (drawn after path so it sits on top)
    if blk_x:
        fig.add_trace(go.Scatter3d(x=blk_x, y=blk_y, z=blk_z, mode='lines',
            line=dict(color='red', width=6, dash='dash'), name='🔥 Blocked'))
    
    # Tunnel
    if tun_x:
        fig.add_trace(go.Scatter3d(x=tun_x, y=tun_y, z=tun_z, mode='lines',
            line=dict(color='orange', width=6, dash='dot'), name='🪜 Ladder'))

    src = st.session_state.source
    destinations = st.session_state.destinations
    
    node_x, node_y, node_z, node_col, node_txt = [], [], [], [], []
    for i in range(n):
        x, y, z = pos[i]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_txt.append(graph.get_node_name(i))
        if i == src: node_col.append('#22c55e')
        elif i in destinations: node_col.append('#a855f7')
        elif i in fire_nodes: node_col.append('#ef4444')
        else: node_col.append('#1e40af')

    fig.add_trace(go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers+text',
        text=node_txt, textposition='top center',
        marker=dict(size=10, color=node_col, line=dict(width=2, color='white')),
        textfont=dict(color="white", size=10), name='Rooms'))

    if active_anim_result and active_anim_result.found():
        path = active_anim_result.path
        start_x, start_y, start_z = pos[path[0]]

        fig.add_trace(go.Scatter3d(x=[start_x], y=[start_y], z=[start_z], mode='lines',
            line=dict(color='white', width=12), name='Your Path'))

        fig.add_trace(go.Scatter3d(x=[start_x], y=[start_y], z=[start_z], mode='markers',
            marker=dict(size=14, color='yellow', symbol='diamond'), name='🏃 You'))

        traces_indices = [len(fig.data) - 2, len(fig.data) - 1]

        frames = []
        for step in range(len(path)):
            node = path[step]
            px, py, pz = pos[node]
            
            trail_x = [pos[path[k]][0] for k in range(step + 1)]
            trail_y = [pos[path[k]][1] for k in range(step + 1)]
            trail_z = [pos[path[k]][2] for k in range(step + 1)]

            frames.append(go.Frame(
                data=[
                    go.Scatter3d(x=trail_x, y=trail_y, z=trail_z),
                    go.Scatter3d(x=[px], y=[py], z=[pz])
                ],
                traces=traces_indices,
                name=f"frame{step}"
            ))
        fig.frames = frames

        fig.update_layout(updatemenus=[dict(type='buttons', showactive=False,
            y=1.0, x=0.0, xanchor='left', yanchor='top',
            buttons=[dict(label='▶ Play Route', method='animate',
                args=[None, dict(frame=dict(duration=1500, redraw=True), transition=dict(duration=800), fromcurrent=True, mode='immediate')])])])

    fig.update_layout(
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, title='', tickvals=[f * 30 for f in floor_groups.keys()], ticktext=[f"Floor {f}" for f in floor_groups.keys()]),
            camera=dict(up=dict(x=0,y=0,z=1), center=dict(x=0,y=0,z=0), eye=dict(x=1.5,y=1.5,z=0.8))
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.3)", font=dict(size=10))
    )
    return fig



# ─────────────────────────────────────────────
# RENDER PATH (step-by-step)
# ─────────────────────────────────────────────
def render_path(result, graph, label, is_emergency, used_tunnel=False):
    css_class = "emergency" if is_emergency else "normal"

    if used_tunnel:
        st.markdown('<div class="tunnel-banner">🪜 Emergency Ladder Route Active — All corridors to exits are blocked. Use the emergency ladder shown on the graph (orange dashed line).</div>',
                    unsafe_allow_html=True)

    if not result.found():
        st.markdown(f'<div class="path-banner blocked">⚠️ {label} — No path found to any exit!</div>',
                    unsafe_allow_html=True)
        return

    exit_name = graph.get_node_name(result.destination_node)
    path_names = [graph.get_node_name(n) for n in result.path]
    arrow_path = " → ".join(path_names)
    icon = "🪜" if used_tunnel else ("🔥" if is_emergency else "🗺️")
    banner_class = "tunnel" if used_tunnel else css_class

    st.markdown(f'<div class="path-banner {banner_class}">{icon} {label} (Target: {exit_name}): {arrow_path}</div>',
                unsafe_allow_html=True)

    # Distance + estimated time (assume 1 unit = 2 metres, walking 1.4 m/s)
    dist = result.distance
    if dist != INT_MAX:
        metres = dist * 2
        seconds = round(metres / 1.4)
        st.markdown(f"**Total distance: `{dist}` units** &nbsp;|&nbsp; ⏱️ Estimated time: **~{seconds} seconds**")
    else:
        st.markdown("**Total distance: N/A**")

    st.markdown("**Step-by-step directions:**")
    for i in range(len(result.path) - 1):
        u = result.path[i]
        v = result.path[i + 1]
        key = (min(u, v), max(u, v))
        wt = st.session_state.original_weights.get(key, "?")

        # Detect staircase step
        floor_u = graph.get_node_floor(u)
        floor_v = graph.get_node_floor(v)
        is_stair = floor_u != floor_v
        stair_icon = " 🪜" if is_stair else ""
        row_class = "stair" if is_stair else ("emergency" if is_emergency else "")
        num_class = "emergency" if is_emergency else ""

        st.markdown(
            f'<div class="step-row {row_class}">'
            f'<span class="step-num {num_class}">{i+1}</span>'
            f'<span>{graph.get_node_name(u)} ──({wt})──▶ {graph.get_node_name(v)}{stair_icon}</span>'
            f'</div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────
# MST GRAPH
# ─────────────────────────────────────────────
def render_mst_graph(graph, mst_edges, mst_cost):
    n = graph.get_number_of_nodes()
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    pos = {}
    floor_groups = {}
    for i in range(n):
        floor = graph.get_node_floor(i)
        floor_groups.setdefault(floor, []).append(i)

    for floor, nodes in floor_groups.items():
        subgraph = nx.subgraph(G, nodes)
        if len(nodes) == 1:
            pos[nodes[0]] = (0, 0, floor * 30)
        else:
            sub_pos = nx.spring_layout(subgraph, k=2.0, iterations=50, seed=42 + floor)
            for node in nodes:
                x, y = sub_pos[node]
                pos[node] = (x * 40, y * 40, floor * 30)

    mst_set = {(min(u, v), max(u, v)) for u, v, *_ in mst_edges}

    mst_x, mst_y, mst_z = [], [], []
    other_x, other_y, other_z = [], [], []

    all_edges = graph.get_all_edges()
    for u, v, w in all_edges:
        if graph.is_blocked(u, v):
            continue
        key = (min(u, v), max(u, v))
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        if key in mst_set:
            mst_x.extend([x0, x1, None])
            mst_y.extend([y0, y1, None])
            mst_z.extend([z0, z1, None])
        else:
            other_x.extend([x0, x1, None])
            other_y.extend([y0, y1, None])
            other_z.extend([z0, z1, None])

    fig = go.Figure()
    if other_x:
        fig.add_trace(go.Scatter3d(
            x=other_x, y=other_y, z=other_z,
            mode='lines',
            line=dict(color='rgba(150, 150, 150, 0.4)', width=2, dash='dash'),
            name='Other Corridors'
        ))
    if mst_x:
        fig.add_trace(go.Scatter3d(
            x=mst_x, y=mst_y, z=mst_z,
            mode='lines',
            line=dict(color='#fbbf24', width=6),
            name='MST Wiring'
        ))

    node_x, node_y, node_z = [], [], []
    node_text = []
    node_color = []
    
    for i in range(n):
        x, y, z = pos[i]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(graph.get_node_name(i))
        node_color.append('#a855f7' if i in st.session_state.destinations else '#1e40af')

    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(size=10, color=node_color, line=dict(width=2, color='white')),
        textfont=dict(color='white', size=10),
        name='Nodes'
    ))

    fig.update_layout(
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, title='', tickvals=[f * 30 for f in floor_groups.keys()], ticktext=[f"Floor {f}" for f in floor_groups.keys()]),
            camera=dict(up=dict(x=0,y=0,z=1), center=dict(x=0,y=0,z=0), eye=dict(x=1.5,y=1.5,z=0.8))
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    st.markdown("### 📂 Upload Building Map")
    uploaded = st.file_uploader("Upload input.txt", type=["txt"])
    if uploaded:
        if st.session_state.get("last_uploaded_file_id") != uploaded.file_id:
            try:
                content = uploaded.getvalue().decode("utf-8")
                g, src, dst, ow = parse_input_file(content)
                st.session_state.graph = g
                st.session_state.source = src
                st.session_state.destinations = dst
                st.session_state.original_weights = ow
                st.session_state.blocked_edges = []
                st.session_state.fire_nodes = []
                st.session_state.last_uploaded_file_id = uploaded.file_id
            except Exception as e:
                st.error(f"❌ Failed to parse file: {e}")

    st.divider()

    if st.button("🔄 Reset to Default Building", use_container_width=True):
        g, src, dst, ow = build_default_graph()
        st.session_state.graph = g
        st.session_state.source = src
        st.session_state.destinations = dst
        st.session_state.original_weights = ow
        st.session_state.blocked_edges = []
        st.session_state.fire_nodes = []
        st.session_state.animate_path = False
        st.session_state.anim_step = 0
        st.rerun()

    st.divider()

    graph = st.session_state.graph
    n = graph.get_number_of_nodes()

    st.markdown("### 🏢 Building Info")
    st.markdown(f"- **Nodes:** {n}")
    st.markdown(f"- **Edges:** {len(graph.get_all_edges())}")
    exit_names = ', '.join([graph.get_node_name(d) for d in st.session_state.destinations])
    st.markdown(f"- **Exits:** {exit_names}")

    st.divider()

    st.markdown("### 📍 Current Location")
    source_options = {graph.get_node_name(i): i for i in range(n)}
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
    if st.session_state.blocked_edges:
        for u, v in st.session_state.blocked_edges:
            st.markdown(f"- `{graph.get_node_name(u)}` ↔ `{graph.get_node_name(v)}`")
    else:
        st.markdown("_No corridors blocked_")

    if st.session_state.blocked_edges:
        if st.button("🧹 Clear All Blockages", use_container_width=True):
            for u, v in st.session_state.blocked_edges:
                key = (min(u, v), max(u, v))
                wt = st.session_state.original_weights.get(key, 1)
                st.session_state.graph.unblock_corridor(u, v, wt)  # actually restore the weight
            st.session_state.blocked_edges = []
            st.session_state.fire_nodes = []
            st.session_state.animate_path = False
            st.session_state.anim_step = 0
            st.rerun()

    st.divider()

    st.markdown("### 🏆 Distance Leaderboard")
    st.caption("Ordered via Merge Sort")
    primary_exit = st.session_state.destinations[0] if st.session_state.destinations else 0
    distances = PathFinder.get_all_distances(graph, primary_exit)
    room_list = [{"name": graph.get_node_name(i), "dist": distances[i]} for i in range(n)]
    sorted_rooms = merge_sort(room_list, key=lambda x: x["dist"])
    for r in sorted_rooms:
        if r["dist"] == 0:
            st.markdown(f"- **{r['name']}**: _Exit_")
        elif r["dist"] == INT_MAX:
            st.markdown(f"- **{r['name']}**: _Unreachable_")
        else:
            st.markdown(f"- **{r['name']}**: `{r['dist']}` pts")


# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────
st.markdown("# 🚨 Emergency Exit Finder")
st.markdown("_Graph-based shortest path with real-time corridor blocking — powered by Dijkstra's algorithm_")
st.divider()

graph = st.session_state.graph
n = graph.get_number_of_nodes()

# ── Build normal graph (no blockages) ──
normal_graph = copy.deepcopy(graph)
for u, v in st.session_state.blocked_edges:
    key = (min(u, v), max(u, v))
    wt = st.session_state.original_weights.get(key, 1)
    normal_graph.unblock_corridor(u, v, wt)

normal_result = PathFinder.find_shortest_path(
    normal_graph, st.session_state.source, st.session_state.destinations)

# ── Build emergency graph (with blockages) ──
emergency_result = None
emergency_graph = None
tunnel_edges_used = []
used_tunnel = False

if st.session_state.blocked_edges:
    emergency_graph = copy.deepcopy(graph)
    emergency_result = PathFinder.find_shortest_path(
        emergency_graph, st.session_state.source, st.session_state.destinations)

    # If emergency path also fails → try tunnel
    if not emergency_result.found():
        g_tunnel = copy.deepcopy(graph)
        tunnel_edges_used = add_tunnel_edges(g_tunnel, st.session_state.destinations, st.session_state.source)
        tunnel_result = PathFinder.find_shortest_path(
            g_tunnel, st.session_state.source, st.session_state.destinations)
        if tunnel_result.found():
            emergency_result = tunnel_result
            used_tunnel = True

# ── Metric Cards ──
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if normal_result.found():
        nd = normal_result.distance
        t_sec = round(nd * 2 / 1.4)
        val_cls = "safe"
    else:
        nd, t_sec, val_cls = "N/A", "-", "danger"
    st.markdown(f'<div class="metric-card"><div class="label">Normal Distance</div>'
                f'<div class="value {val_cls}">{nd}</div></div>', unsafe_allow_html=True)

with col2:
    if emergency_result:
        if emergency_result.found():
            ed = emergency_result.distance
            val_cls = "warn" if used_tunnel else "safe"
        else:
            ed, val_cls = "N/A", "danger"
    else:
        ed, val_cls = "—", "safe"
    st.markdown(f'<div class="metric-card"><div class="label">Emergency Distance</div>'
                f'<div class="value {val_cls}">{ed}</div></div>', unsafe_allow_html=True)

with col3:
    blk = len(st.session_state.blocked_edges)
    blk_cls = "danger" if blk > 0 else "safe"
    st.markdown(f'<div class="metric-card"><div class="label">Blocked Corridors</div>'
                f'<div class="value {blk_cls}">{blk}</div></div>', unsafe_allow_html=True)

with col4:
    hops = (len(normal_result.path) - 1) if normal_result.found() else 0
    st.markdown(f'<div class="metric-card"><div class="label">Normal Path Hops</div>'
                f'<div class="value">{hops}</div></div>', unsafe_allow_html=True)

with col5:
    if normal_result.found():
        t_sec = round(normal_result.distance * 2 / 1.4)
        st.markdown(f'<div class="metric-card"><div class="label">Est. Evac Time</div>'
                    f'<div class="value safe">{t_sec}s</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="metric-card"><div class="label">Est. Evac Time</div>'
                    f'<div class="value danger">N/A</div></div>', unsafe_allow_html=True)

st.divider()

# ── Tunnel warning banner ──
if used_tunnel:
    st.markdown('<div class="tunnel-banner">🪜 ⚠️ All normal exits are blocked! An emergency ladder route has been activated from your location directly to the nearest exit. Follow the orange dashed path on the graph above.</div>',
                unsafe_allow_html=True)

# ── Main layout: Graph + Controls ──
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### 🗺️ Building Graph")

    # Determine active path (emergency if blocked, else normal)
    if st.session_state.blocked_edges and emergency_result and emergency_result.found():
        active_anim_result = emergency_result
    elif normal_result.found():
        active_anim_result = normal_result
    else:
        active_anim_result = None

    fig = draw_graph(
        graph, normal_result, emergency_result,
        st.session_state.blocked_edges, st.session_state.fire_nodes,
        tunnel_edges=tunnel_edges_used if used_tunnel else None,
        active_anim_result=active_anim_result,
    )
    st.plotly_chart(fig, use_container_width=True, config={'modeBarButtonsToRemove': ['toImage', 'orbitRotation', 'tableRotation', 'resetCameraLastSave3d']})

with right:
    # ── Block Corridor ──
    st.markdown("### 🔥 Simulate Fire / Block Corridor")
    edges = graph.get_all_edges()
    edge_options = {}
    for u, v, w in edges:
        label = f"{graph.get_node_name(u)} ↔ {graph.get_node_name(v)} (dist {w})"
        edge_options[label] = (u, v, w)

    if edge_options:
        selected_label = st.selectbox("Select corridor to block", list(edge_options.keys()))
        if st.button("🔥 Block this corridor", use_container_width=True, type="primary"):
            u, v, w = edge_options[selected_label]
            key = (min(u, v), max(u, v))
            blocked_keys = [(min(a, b), max(a, b)) for a, b in st.session_state.blocked_edges]
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
        u, v = int(n1), int(n2)
        if u == v:
            st.error("Cannot block a node with itself.")
        elif graph.get_edge_weight(u, v) == 0:
            st.error(f"No corridor exists between {graph.get_node_name(u)} and {graph.get_node_name(v)}.")
        else:
            key = (min(u, v), max(u, v))
            blocked_keys = [(min(a, b), max(a, b)) for a, b in st.session_state.blocked_edges]
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

    # ── Adjacency Matrix ──
    with st.expander("🧮 Adjacency Matrix", expanded=False):
        import pandas as pd
        names = [graph.get_node_name(i) for i in range(n)]
        matrix_data = []
        for i in range(n):
            row = []
            for j in range(n):
                w = graph.adjacency_matrix[i][j]
                row.append("🔥" if w == INT_MAX else ("-" if w == 0 else str(w)))
            matrix_data.append(row)
        df = pd.DataFrame(matrix_data, index=names, columns=names)
        st.dataframe(df, use_container_width=True)


# ─────────────────────────────────────────────
# TABS: Routing + MST
# ─────────────────────────────────────────────
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
        if st.session_state.blocked_edges:
            render_path(emergency_result, graph, "Re-routed → EXIT",
                        is_emergency=not used_tunnel, used_tunnel=used_tunnel)
            if normal_result.found() and emergency_result and emergency_result.found() and not used_tunnel:
                diff = emergency_result.distance - normal_result.distance
                if diff > 0:
                    st.markdown(f"⚠️ **Detour adds `{diff}` extra distance units** due to blockage.")
                elif diff == 0:
                    st.markdown("✅ **Emergency route has the same distance as normal.**")
        else:
            st.info("Block a corridor above to see the emergency re-routing.")

    st.divider()
    st.markdown("### 📊 Algorithm Comparison")
    st.caption("Comparing Pathfinding algorithms in the **current building state**.")

    g_target = emergency_graph if st.session_state.blocked_edges and emergency_graph else normal_graph
    target_result = emergency_result if st.session_state.blocked_edges and emergency_result else normal_result

    bfs_result = PathFinder.find_path_bfs(g_target, st.session_state.source, st.session_state.destinations)
    dfs_result = PathFinder.find_path_dfs(g_target, st.session_state.source, st.session_state.destinations)

    def get_algo_stats(name, result, g):
        if not result.found():
            return {"Algorithm": name, "Status": "No Path", "Weight Cost": "-",
                    "Hops": "-", "Nodes Explored": str(result.visited_count)}
        cost = sum(g.get_edge_weight(result.path[i], result.path[i+1])
                   for i in range(len(result.path)-1))
        hops = len(result.path) - 1
        path_str = " → ".join([g.get_node_name(x) for x in result.path])
        return {"Algorithm": name, "Status": "Path Found", "Weight Cost": str(cost),
                "Hops": str(hops), "Nodes Explored": str(result.visited_count), "Path": path_str}

    import pandas as pd
    comp_data = [
        get_algo_stats("Dijkstra (Weighted)", target_result, g_target),
        get_algo_stats("BFS (Unweighted)", bfs_result, g_target),
        get_algo_stats("DFS (Unweighted)", dfs_result, g_target),
    ]
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

with tab2:
    st.markdown("## 💡 Minimum Spanning Tree (Alarm Wiring)")
    st.markdown("Using **Kruskal's Algorithm**, we calculate the cheapest way to lay fire alarm wiring so all rooms are connected without redundant loops.")
    g_target = emergency_graph if st.session_state.blocked_edges and emergency_graph else normal_graph
    mst_edges, mst_cost = kruskal_mst(g_target)
    st.success(f"**Optimal Wiring Cost:** `{mst_cost}` units")
    fig = render_mst_graph(g_target, mst_edges, mst_cost)
    st.plotly_chart(fig, use_container_width=True, config={'modeBarButtonsToRemove': ['toImage', 'orbitRotation', 'tableRotation', 'resetCameraLastSave3d']})
