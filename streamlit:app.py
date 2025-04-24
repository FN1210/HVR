import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import nolds

# ---------- App Config ----------
st.set_page_config(page_title="HRV Visualisierung", layout="centered")
st.title("🫀 HRV Analyse: Poincaré Plot, Visibility Graph & DFA")

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
        marker=dict(size=5, color='rgba(0,100,255,0.5)'),
        name='RR[n] vs RR[n+1]'
    ))

    fig.add_trace(go.Scatter(
        x=[min(x), max(x)],
        y=[min(x), max(x)],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Identity line'
    ))

    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = mean_rr + SD2 * np.cos(theta) * np.cos(np.pi/4) - SD1 * np.sin(theta) * np.sin(np.pi/4)
    ellipse_y = mean_rr + SD2 * np.cos(theta) * np.sin(np.pi/4) + SD1 * np.sin(theta) * np.cos(np.pi/4)

    fig.add_trace(go.Scatter(
        x=ellipse_x, y=ellipse_y,
        mode='lines',
        line=dict(color='yellow', dash='dot'),
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
    if len(degrees) > 0:
        plt.figure(figsize=(8, 4))
        plt.hist(degrees, bins=range(min(degrees), max(degrees)+1), alpha=0.8, edgecolor='black')
        plt.title("Degree Distribution of Visibility Graph")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.warning("Nicht genug Knoten im Visibility Graph für eine Verteilung.")

# ---------- DFA Analyse ----------
def compute_dfa(rr):
    alpha = nolds.dfa(rr)
    return alpha

def plot_dfa_loglog(rr):
    rr = rr - np.mean(rr)
    nvals = np.unique(np.logspace(1, np.log10(len(rr)//4), num=20, dtype=int))
    F_n = []

    for n in nvals:
        segments = len(rr) // n
        if segments < 2:
            continue
        reshaped = rr[:segments * n].reshape((segments, n))
        local_trends = np.polyfit(np.arange(n), reshaped.T, deg=1)
        detrended = reshaped - (local_trends[0] * np.arange(n) + local_trends[1])
        fluct = np.sqrt(np.mean(detrended**2, axis=1))
        F_n.append(np.mean(fluct))

    plt.figure()
    plt.plot(nvals[:len(F_n)], F_n, 'o-', label='F(n) vs n')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Fenstergröße n (log)')
    plt.ylabel('Fluktuation F(n) (log)')
    plt.title('DFA – Log-Log-Darstellung')
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

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

    st.subheader("📉 DFA – Detrended Fluctuation Analysis")
    alpha = compute_dfa(rr_intervals)
    st.success(f"✅ **DFA α-Wert**: {alpha:.3f}")

    if alpha < 0.7:
        st.info("🔎 Geringe Korrelation (möglicherweise zufällige HRV-Muster).")
    elif 0.7 <= alpha <= 1.0:
        st.info("✅ Gesunde, komplexe HRV-Struktur.")
    else:
        st.info("⚠️ Möglicherweise pathologische HRV-Struktur (übermäßig reguliert).")

    plot_dfa_loglog(rr_intervals)

else:
    st.info("Bitte lade eine .txt-Datei mit RR-Intervallen hoch (eine Zahl pro Zeile).")
