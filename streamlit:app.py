import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import nolds
from io import StringIO

# ---------- App Config ----------
st.set_page_config(page_title="HRV Visualisierung", layout="centered")
st.title("🫀 HRV Analyse: Poincaré Plot, Visibility Graph & DFA")

# ---------- Helper Functions ----------
def validate_rr_intervals(rr):
    """Validate RR intervals data"""
    if len(rr) < 10:
        st.error("⚠️ Zu wenige RR-Intervalle für eine aussagekräftige Analyse (mindestens 10 benötigt).")
        return False
    if any(rr <= 0):
        st.error("⚠️ RR-Intervalle müssen positive Werte sein.")
        return False
    if np.std(rr) == 0:
        st.error("⚠️ RR-Intervalle zeigen keine Variabilität.")
        return False
    return True

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

    # Main scatter plot
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=8, color='rgba(70, 130, 180, 0.7)', line=dict(width=1, color='DarkSlateGrey')),
        name='RR[n] vs RR[n+1]',
        hovertemplate='RR[n]: %{x:.2f} ms<br>RR[n+1]: %{y:.2f} ms<extra></extra>'
    ))

    # Identity line
    fig.add_trace(go.Scatter(
        x=[min(x), max(x)],
        y=[min(x), max(x)],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Identitätslinie'
    ))

    # SD1/SD2 ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = mean_rr + SD2 * np.cos(theta) * np.cos(np.pi/4) - SD1 * np.sin(theta) * np.sin(np.pi/4)
    ellipse_y = mean_rr + SD2 * np.cos(theta) * np.sin(np.pi/4) + SD1 * np.sin(theta) * np.cos(np.pi/4)

    fig.add_trace(go.Scatter(
        x=ellipse_x, y=ellipse_y,
        mode='lines',
        line=dict(color='blue', width=2),
        name=f'Ellipse (SD1: {SD1:.2f}, SD2: {SD2:.2f})',
        fill='toself',
        fillcolor='rgba(100, 149, 237, 0.2)'
    ))

    fig.update_layout(
        title="Poincaré Plot mit SD1 & SD2",
        xaxis_title="RR[n] (ms)",
        yaxis_title="RR[n+1] (ms)",
        width=700,
        height=700,
        margin=dict(l=60, r=60, t=80, b=60),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='closest'
    )

    fig.update_xaxes(range=[min(x)*0.95, max(x)*1.05])
    fig.update_yaxes(range=[min(y)*0.95, max(y)*1.05])
    
    st.plotly_chart(fig, use_container_width=True)
    return SD1, SD2

# ---------- Visibility Graph ----------
def visibility_graph(ts):
    G = nx.Graph()
    N = len(ts)
    G.add_nodes_from(range(N))
    
    for i in range(N):
        for j in range(i+1, N):
            visible = True
            for k in range(i+1, j):
                if ts[k] >= ts[i] + (ts[j] - ts[i]) * (k - i) / (j - i):
                    visible = False
                    break
            if visible:
                G.add_edge(i, j)
    return G

def plot_visibility_graph(G, rr):
    degrees = [deg for _, deg in G.degree()]
    
    if len(degrees) < 2:
        st.warning("Nicht genug Verbindungen im Visibility Graph für eine aussagekräftige Analyse.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Degree distribution
    ax1.hist(degrees, bins=np.arange(min(degrees)-0.5, max(degrees)+1.5, 1), 
            alpha=0.7, edgecolor='black', density=True)
    ax1.set_title("Knotengrad-Verteilung")
    ax1.set_xlabel("Grad")
    ax1.set_ylabel("Relative Häufigkeit")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Graph visualization (simplified)
    pos = {i: (i, rr[i]) for i in range(len(rr))}
    nx.draw_networkx(G, pos, ax=ax2, node_size=20, width=0.5, 
                    with_labels=False, alpha=0.7)
    ax2.set_title("Visibility Graph (vereinfacht)")
    ax2.set_xlabel("Zeitindex")
    ax2.set_ylabel("RR-Intervall (ms)")
    
    plt.tight_layout()
    st.pyplot(fig)

# ---------- DFA Analyse ----------
def compute_dfa(rr):
    try:
        alpha = nolds.dfa(rr)
        return alpha
    except Exception as e:
        st.error(f"Fehler bei DFA-Berechnung: {str(e)}")
        return None

def plot_dfa_loglog(rr):
    rr = rr - np.mean(rr)
    nvals = np.unique(np.logspace(np.log10(4), np.log10(len(rr)//4), num=20, dtype=int))
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
    
    if len(F_n) < 3:
        st.warning("Nicht genug Datenpunkte für DFA-Log-Log-Plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(nvals[:len(F_n)], F_n, 'o-', markersize=8, label='F(n) vs n')
    
    # Add regression line
    coeffs = np.polyfit(np.log(nvals[:len(F_n)]), np.log(F_n), 1)
    regression_line = np.exp(coeffs[1]) * nvals[:len(F_n)]**coeffs[0]
    ax.loglog(nvals[:len(F_n)], regression_line, 'r--', 
             label=f'Regressionsgerade (α={coeffs[0]:.2f})')
    
    ax.set_xlabel('Fenstergröße n (log)', fontsize=12)
    ax.set_ylabel('Fluktuation F(n) (log)', fontsize=12)
    ax.set_title('DFA – Log-Log-Darstellung', fontsize=14)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

# ---------- Hauptlogik ----------
if uploaded_file is not None:
    try:
        content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        rr_intervals = np.array([float(line.strip()) for line in content.splitlines() if line.strip()])
        
        if not validate_rr_intervals(rr_intervals):
            st.stop()
            
        with st.expander("📊 RR-Intervalle Vorschau", expanded=False):
            st.dataframe(rr_intervals, height=200)
            st.write(f"Anzahl der RR-Intervalle: {len(rr_intervals)}")
            st.write(f"Mittelwert: {np.mean(rr_intervals):.2f} ± {np.std(rr_intervals):.2f} ms")
        
        # Analysis sections
        st.subheader("📈 Poincaré Plot")
        with st.spinner("Berechne Poincaré Plot..."):
            sd1, sd2 = plot_poincare_plotly(rr_intervals)
            col1, col2 = st.columns(2)
            col1.metric("SD1 (kurzfristige Variabilität)", f"{sd1:.2f} ms")
            col2.metric("SD2 (langfristige Variabilität)", f"{sd2:.2f} ms")
            st.write("""
            - **SD1**: Repräsentiert die kurzfristige HRV (parasympathische Aktivität)
            - **SD2**: Repräsentiert die langfristige HRV (sympathische und parasympathische Aktivität)
            """)
        
        st.subheader("🌐 Visibility Graph Analyse")
        with st.spinner("Erstelle Visibility Graph..."):
            G = visibility_graph(rr_intervals)
            plot_visibility_graph(G, rr_intervals)
            st.write("""
            Der Visibility Graph wandelt die Zeitreihe in ein Netzwerk um, wo ähnliche Werte verbunden sind.
            Eine skalierungsfreie Verteilung der Knotengrade deutet auf fraktale Eigenschaften hin.
            """)
        
        st.subheader("📉 DFA – Detrended Fluctuation Analysis")
        with st.spinner("Berechne DFA..."):
            alpha = compute_dfa(rr_intervals)
            if alpha is not None:
                st.metric("DFA α-Wert", f"{alpha:.3f}")
                
                if alpha < 0.75:
                    interpretation = "Geringe Korrelation (white noise-ähnlich)"
                elif 0.75 <= alpha < 1.05:
                    interpretation = "Gesunde HRV (1/f-ähnliches Rauschen)"
                elif 1.05 <= alpha < 1.25:
                    interpretation = "Erhöhte Regelmäßigkeit"
                else:
                    interpretation = "Stark korrelierte/regulierte HRV"
                
                st.info(f"**Interpretation**: {interpretation}")
                plot_dfa_loglog(rr_intervals)
                
    except Exception as e:
        st.error(f"Fehler bei der Verarbeitung der Datei: {str(e)}")
        st.stop()
else:
    st.info("""
    **Anleitung**: Bitte lade eine Textdatei mit RR-Intervallen hoch (eine Zahl pro Zeile).
    """)
