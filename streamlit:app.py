import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import nolds

# ---------- App Config ----------
st.set_page_config(page_title="HRV Visualisierung", layout="centered")
st.title("🫀 HRV Analyse: Poincaré Plot, Visibility Graph, GHVE & DFA")

# ---------- Datei-Upload ----------
uploaded_file = st.file_uploader("📤 Lade deine RR-Intervall-Datei (.txt) hoch", type=["txt"])

# ---------- SD1 & SD2 Berechnung ----------
def compute_sd1_sd2(rr):
    diff = rr[1:] - rr[:-1]
    SDSD = np.std(diff, ddof=1)
    SDRR = np.std(rr, ddof=1)
    SD1 = SDSD / np.sqrt(2)
    SD2 = np.sqrt(2 * SDRR**2 - SD1**2)
    return SD1, SD2

# ---------- Poincaré Plot ----------
def plot_poincare_plotly(rr):
    x = rr[:-1]
    y = rr[1:]
    SD1, SD2 = compute_sd1_sd2(rr)
    mean_rr = np.mean(rr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=4, color='#6DCFF6', opacity=0.6),
        name='RR[n] vs RR[n+1]'
    ))
    fig.add_trace(go.Scatter(
        x=[min(x), max(x)],
        y=[min(x), max(x)],
        mode='lines',
        line=dict(color='#002654', dash='dash'),
        name='Identity line'
    ))
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = mean_rr + SD2 * np.cos(theta) * np.cos(np.pi/4) - SD1 * np.sin(theta) * np.sin(np.pi/4)
    ellipse_y = mean_rr + SD2 * np.cos(theta) * np.sin(np.pi/4) + SD1 * np.sin(theta) * np.cos(np.pi/4)
    fig.add_trace(go.Scatter(
        x=ellipse_x, y=ellipse_y,
        mode='lines',
        line=dict(color='#FFFFFF', dash='dot'),
        name='Ellipse (SD1/SD2)'
    ))
    fig.update_layout(
        title="Poincaré Plot",
        xaxis_title="RR[n] (ms)",
        yaxis_title="RR[n+1] (ms)",
        width=700,
        height=700,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True,
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='#FFFFFF'),
        legend=dict(bgcolor='#000000')
    )
    st.plotly_chart(fig, use_container_width=True)
    return SD1, SD2

# ---------- Visibility Graph ----------
def visibility_graph_fast(ts):
    G = nx.Graph()
    N = len(ts)
    G.add_nodes_from(range(N))
    for i in range(N):
        t_i, y_i = i, ts[i]
        for j in range(i+1, N):
            t_j, y_j = j, ts[j]
            slope = (y_j - y_i) / (t_j - t_i)
            visible = True
            for k in range(i+1, j):
                t_k, y_k = k, ts[k]
                y_interp = y_i + slope * (t_k - t_i)
                if y_k >= y_interp:
                    visible = False
                    break
            if visible:
                G.add_edge(i, j)
    return G

# ---------- GHVE (schnell) ----------
def ghve_visibility_graph_fast(rr_diff):
    N = len(rr_diff)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N-1):
        max_val = rr_diff[i]
        for j in range(i+1, N):
            if rr_diff[j] < max_val:
                continue
            G.add_edge(i, j)
            max_val = rr_diff[j]
    return G

# ---------- Entropie ----------
def compute_ghve_entropy(G):
    degrees = np.array([d for n, d in G.degree()])
    if len(degrees) == 0:
        return np.nan
    hist = np.bincount(degrees)
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log(prob))
    return entropy

# ---------- Plots ----------
def plot_visibility_graph(G):
    degrees = [deg for _, deg in G.degree()]
    if degrees:
        plt.figure(figsize=(8, 4), facecolor='#000')
        plt.hist(degrees, bins=range(min(degrees), max(degrees)+1), alpha=0.8, color='#6DCFF6', edgecolor='white')
        plt.title("Degree Distribution of Visibility Graph", color='white')
        plt.xlabel("Degree", color='white')
        plt.ylabel("Frequency", color='white')
        plt.grid(True, color='#555', linestyle='--', linewidth=0.5)
        plt.gca().tick_params(colors='white')
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        st.pyplot(plt.gcf())
        plt.close()

def plot_visibility_network(G):
    if len(G.nodes) > 0:
        plt.figure(figsize=(10, 6), facecolor='#000')
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color='#6DCFF6', alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='#FFFFFF', alpha=0.2)
        plt.title("Graphisches Netzwerkschaubild", color='white')
        plt.axis('off')
        plt.gca().set_facecolor('#000')
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.warning("Nicht genug Knoten für Netzwerkvisualisierung.")

def plot_ghve(rr):
    rr_diff = np.diff(rr)
    plt.figure(figsize=(8, 4), facecolor='#000')
    plt.plot(rr_diff, label='RR Differenzen', color='#6DCFF6')
    plt.title('GHVE – Gradient Horizontal Visibility Edges', color='white')
    plt.xlabel('Index', color='white')
    plt.ylabel('Differenz RR[n+1] - RR[n] (ms)', color='white')
    plt.grid(True, color='#555', linestyle='--', linewidth=0.5)
    plt.legend(facecolor='#000', edgecolor='#FFF', labelcolor='white')
    plt.gca().tick_params(colors='white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    st.pyplot(plt.gcf())
    plt.close()

# ---------- DFA ----------
def compute_dfa(rr):
    return nolds.dfa(rr)

def plot_dfa_loglog(rr):
    rr = rr - np.mean(rr)
    nvals = np.unique(np.logspace(1, np.log10(len(rr)//4), num=20, dtype=int))
    F_n = []

    for n in nvals:
        segments = len(rr) // n
        if segments < 2:
            continue
        reshaped = rr[:segments * n].reshape((segments, n))
        F_seg = []
        for seg in reshaped:
            p = np.polyfit(np.arange(n), seg, 1)
            trend = p[0] * np.arange(n) + p[1]
            detrended = seg - trend
            fluct = np.sqrt(np.mean(detrended ** 2))
            F_seg.append(fluct)
        F_n.append(np.mean(F_seg))

    plt.figure(facecolor='#000')
    plt.plot(nvals[:len(F_n)], F_n, 'o-', label='F(n) vs n', color='#6DCFF6')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Fenstergröße n (log)', color='white')
    plt.ylabel('Fluktuation F(n) (log)', color='white')
    plt.title('DFA – Log-Log-Darstellung', color='white')
    plt.grid(True, which="both", ls="--", lw=0.5, color='#555')
    plt.legend(facecolor='#000', edgecolor='#FFF', labelcolor='white')
    plt.gca().tick_params(colors='white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    st.pyplot(plt.gcf())
    plt.close()

# ---------- Hauptlogik ----------
if uploaded_file is not None:
    rr_lines = uploaded_file.read().decode("utf-8").splitlines()
    rr_intervals = np.array([float(line.strip()) for line in rr_lines if line.strip()])

    st.subheader("📈 Poincaré Plot")
    sd1, sd2 = plot_poincare_plotly(rr_intervals)
    st.success(f"✅ **SD1** (kurzfristige HRV): {sd1:.2f} ms")
    st.success(f"✅ **SD2** (langfristige HRV): {sd2:.2f} ms")

    st.subheader("🌐 Visibility Graph Analyse")
    with st.spinner("Erzeuge Visibility Graph..."):
        G = visibility_graph_fast(rr_intervals[:1000])
    plot_visibility_graph(G)

    st.subheader("🌐 Sichtbarkeitsgraph (Netzwerk)")
    plot_visibility_network(G)

    st.subheader("📊 GHVE – Gradient Horizontal Visibility Edges")
    plot_ghve(rr_intervals)

    st.subheader("🌐 GHVE Netzwerk und Entropie")
    with st.spinner("Berechne GHVE Netzwerk..."):
        rr_diff = np.diff(rr_intervals)
        G_ghve = ghve_visibility_graph_fast(rr_diff)
        ghve_entropy = compute_ghve_entropy(G_ghve)
    st.success(f"✅ **GHVE Entropie**: {ghve_entropy:.3f}")
    plot_visibility_network(G_ghve)

    st.subheader("📉 DFA – Detrended Fluctuation Analysis")
    alpha = compute_dfa(rr_intervals)
    st.success(f"✅ **DFA α-Wert**: {alpha:.3f}")
    plot_dfa_loglog(rr_intervals)
else:
    st.info("Bitte lade eine .txt-Datei mit RR-Intervallen hoch (eine Zahl pro Zeile).")
