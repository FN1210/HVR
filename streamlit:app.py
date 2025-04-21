import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------- App Config ----------
st.set_page_config(page_title="HRV Visualisierung", layout="centered")
st.title("🫀 HRV Analyse: Poincaré Plot & Visibility Graph")

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

# ---------- Interaktiver Poincaré Plot mit Plotly ----------
def plot_poincare_plotly(rr):
    x = rr[:-1]
    y = rr[1:]
    SD1, SD2 = compute_sd1_sd2(rr)
    mean_rr = np.mean(rr)

    fig = go.Figure()

    # Punktwolke
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=5, color='rgba(0,100,255,0.5)'),
        name='RR[n] vs RR[n+1]'
    ))

    # Identitätslinie
    fig.add_trace(go.Scatter(
        x=[min(x), max(x)],
        y=[min(x), max(x)],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Identity line'
    ))

    # Ellipse (SD1/SD2)
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = mean_rr + SD2 * np.cos(theta) * np.cos(np.pi/4) - SD1 * np.sin(theta) * np.sin(np.pi/4)
    ellipse_y = mean_rr + SD2 * np.cos(theta) * np.sin(np.pi/4) + SD1 * np.sin(theta) * np.cos(np.pi/4)

    fig.add_trace(go.Scatter(
        x=ellipse_x, y=ellipse_y,
        mode='lines',
        line=dict(color='blue', dash='dot'),
        name='Ellipse (SD1/SD2)'
    ))

    fig.update_layout(
        title="Poincaré Plot mit SD1 & SD2 (Zoom-fähig)",
        xaxis_title="RR[n] (ms)",
        yaxis_title="RR[n+1] (ms)",
        width=700,
        height=700,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)
    return SD1, SD2

# ---------- Visibility Graph ----------
def visibility_graph(ts):
    G = nx.Graph()
    N = len(ts)
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i+1, N):
            if all(ts[k] < ts[i] + (ts[j] - ts[i]) * (k - i) / (j - i) for k in range(i+1, j)):
                G.add_edge(i, j)
    return G

def plot_visibility_graph(G):
    degrees = [deg for _, deg in G.degree()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(degrees, bins=range(min(degrees), max(degrees)+1), alpha=0.8, edgecolor='black')
    ax.set_title("Degree Distribution of Visibility Graph")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    st.pyplot(fig)

# ---------- Hauptlogik ----------
if uploaded_file is not None:
    rr_intervals = np.array([float(line.strip()) for line in uploaded_file if line.strip()])

    st.subheader("📈 Poincaré Plot")
    sd1, sd2 = plot_poincare_plotly(rr_intervals)
    st.success(f"✅ **SD1** (kurzfristige HRV): {sd1:.2f} ms")
    st.success(f"✅ **SD2** (langfristige HRV): {sd2:.2f} ms")

    st.subheader("🌐 Visibility Graph Analyse")
    G = visibility_graph(rr_intervals)
    plot_visibility_graph(G)

else:
    st.info("Bitte lade eine .txt-Datei mit RR-Intervallen hoch (eine Zahl pro Zeile).")
