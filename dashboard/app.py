import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from sklearn.linear_model import Ridge

# 1. Konfigurasi Halaman & Judul
st.set_page_config(page_title="Crypto & Stock Market Predictive BI Dashboard", layout="wide")
st.title("📈 Real-Time Big Data & Machine Learning: Prediksi & BI Pasar Kripto-Saham")
st.caption("Sumber Data: Binance Public Data & Indeks Makro/Saham (1 Tahun) | Arsitektur: Hadoop HDFS → Apache Spark → PostgreSQL → Real-Time ML")

# 2. Load Data dari PostgreSQL & Ekstrak Parameter Model ML Sekali
@st.cache_resource
def load_data_and_model():
    engine = create_engine("postgresql://bigdata_user:bigdata_pass@localhost:5432/market_dw")
    df = pd.read_sql("SELECT * FROM fact_market_analytics", engine)
    bi_df = pd.read_sql("SELECT * FROM dim_bi_factors", engine)

    features = ["btc_close", "eth_close", "sol_close", "btc_ret_lag1", "eth_ret_lag1", "sol_ret_lag1"]
    model = Ridge(alpha=1.0)
    model.fit(df[features], df["target_next_day_close"])

    model_params = {
        "intercept": float(model.intercept_),
        "coef_btc": float(model.coef_[0]),
        "coef_eth": float(model.coef_[1]),
        "coef_sol": float(model.coef_[2]),
        "coef_btc_chg": float(model.coef_[3]),
        "coef_eth_chg": float(model.coef_[4]),
        "coef_sol_chg": float(model.coef_[5]),
        "top_bi_factor": str(bi_df.iloc[0]["faktor_lampau"]),
        "top_bi_score": float(bi_df.iloc[0]["skor_pengaruh_persen"]),
        "fallback_btc": float(df["btc_close"].iloc[-1]),
        "fallback_eth": float(df["eth_close"].iloc[-1]),
        "fallback_sol": float(df["sol_close"].iloc[-1]),
    }
    return df, bi_df, model_params

df, bi_df, params = load_data_and_model()

# 3. KOMPONEN GRAFIK & PREDIKSI REAL-TIME (SLIDING WINDOW 15 TITIK, UPDATE 5 DETIK)
realtime_widget_html = f"""
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body {{
      background-color: #0e1117;
      color: #fafafa;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 0;
    }}
    .status-bar {{
      font-size: 13px;
      color: #a3a8b8;
      margin-bottom: 14px;
    }}
    .badge {{
      background: #172b24;
      color: #00cc96;
      padding: 3px 8px;
      border-radius: 4px;
      font-family: monospace;
      font-weight: 600;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 16px;
    }}
    .kpi-card {{
      background: #161a24;
      border: 1px solid #262c3a;
      border-radius: 10px;
      padding: 14px 18px;
    }}
    .kpi-title {{
      font-size: 13px;
      color: #a3a8b8;
      margin-bottom: 6px;
    }}
    .kpi-value {{
      font-size: 26px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .kpi-delta {{
      display: inline-block;
      font-size: 12px;
      padding: 2px 8px;
      border-radius: 12px;
      font-weight: 600;
    }}
    .up {{ background: rgba(0, 204, 150, 0.15); color: #00cc96; }}
    .down {{ background: rgba(239, 85, 59, 0.15); color: #ef553b; }}
    .neutral {{ background: rgba(255, 161, 90, 0.15); color: #ffa15a; }}
    #liveChart {{
      width: 100%;
      height: 360px;
      background: #0e1117;
    }}
  </style>
</head>
<body>
  <div class="status-bar">
    Status Aliran Data: <span class="badge" id="connStatus">🟢 LIVE STREAMING (Smooth Real-Time 2 Detik)</span>
    &nbsp;|&nbsp; Pembaruan Terakhir: <span class="badge" id="lastTime">--:--:-- WIB</span>
  </div>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-title">Harga BTC Real-Time</div>
      <div class="kpi-value" id="btcPrice">$--</div>
      <div class="kpi-delta" id="btcDelta">--</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Prediksi ML Berikutnya</div>
      <div class="kpi-value" id="predPrice">$--</div>
      <div class="kpi-delta" id="predDelta">--</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Sinyal Keputusan ML</div>
      <div class="kpi-value" id="mlSignal">--</div>
      <div class="kpi-delta" id="mlMomentum">Momentum: 0.000%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Faktor Penggerak Aktif</div>
      <div class="kpi-value" id="activeDriver">--</div>
      <div class="kpi-delta up" id="biTop">Top BI: {params['top_bi_score']}%</div>
    </div>
  </div>

  <div id="liveChart"></div>

  <script>
    const MODEL = {json.dumps(params)};
    const MAX_POINTS = 15; // Jumlah maksimum titik waktu tampil; titik paling kiri otomatis hilang setelah 15 titik
    let priceHistory = [];

    const traceActual = {{
      x: [],
      y: [],
      mode: 'lines+markers',
      name: 'Harga Aktual Live (USD)',
      line: {{ color: '#00CC96', width: 3 }},
      marker: {{ size: 6 }}
    }};

    const tracePred = {{
      x: [],
      y: [],
      mode: 'lines+markers',
      name: 'Prediksi ML Live (USD)',
      line: {{ color: '#FFA15A', width: 2, dash: 'dash' }},
      marker: {{ size: 5 }}
    }};

    const layout = {{
      title: {{ text: 'Live Streaming Ticker & Prediksi Machine Learning (Auto-Update Setiap 2 Detik)', font: {{ color: '#fafafa', size: 15 }} }},
      paper_bgcolor: '#0e1117',
      plot_bgcolor: '#0e1117',
      xaxis: {{
        title: 'Waktu (WIB)',
        type: 'category',
        color: '#a3a8b8',
        gridcolor: '#1f2430',
        tickangle: 0
      }},
      yaxis: {{
        title: 'Harga (USD)',
        color: '#a3a8b8',
        gridcolor: '#1f2430',
        tickprefix: '$'
      }},
      legend: {{ orientation: 'h', y: 1.12, x: 1, xanchor: 'right', font: {{ color: '#fafafa' }} }},
      margin: {{ t: 45, b: 40, l: 65, r: 20 }}
    }};

    Plotly.newPlot('liveChart', [traceActual, tracePred], layout, {{ responsive: true }});

    function formatUSD(val) {{
      return '$' + val.toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
    }}

    async function updateStream() {{
      const now = new Date();
      const timeStr = now.toTimeString().split(' ')[0];

      let btc, btcChg, eth, ethChg, sol, solChg;

      try {{
        const [rBtc, rEth, rSol] = await Promise.all([
          fetch('https://data-api.binance.vision/api/v3/ticker/24hr?symbol=BTCUSDT'),
          fetch('https://data-api.binance.vision/api/v3/ticker/24hr?symbol=ETHUSDT'),
          fetch('https://data-api.binance.vision/api/v3/ticker/24hr?symbol=SOLUSDT')
        ]);
        const dBtc = await rBtc.json();
        const dEth = await rEth.json();
        const dSol = await rSol.json();

        btc = parseFloat(dBtc.lastPrice);
        btcChg = parseFloat(dBtc.priceChangePercent);
        eth = parseFloat(dEth.lastPrice);
        ethChg = parseFloat(dEth.priceChangePercent);
        sol = parseFloat(dSol.lastPrice);
        solChg = parseFloat(dSol.priceChangePercent);
        document.getElementById('connStatus').innerText = '🟢 LIVE STREAMING (Smooth Real-Time 5 Detik)';
      }} catch (e) {{
        const prev = priceHistory.length > 0 ? priceHistory[priceHistory.length - 1] : MODEL.fallback_btc;
        btc = prev * (1 + (Math.random() - 0.5) * 0.0004);
        btcChg = -2.05;
        eth = MODEL.fallback_eth;
        ethChg = -1.80;
        sol = MODEL.fallback_sol;
        solChg = -3.10;
        document.getElementById('connStatus').innerText = '🟡 SIMULASI STREAMING (Offline Mode)';
      }}

      priceHistory.push(btc);
      if (priceHistory.length > MAX_POINTS) priceHistory.shift();

      // Hitung momentum 5 titik terakhir
      const recent = priceHistory.slice(-5);
      const shortMA = recent.reduce((a, b) => a + b, 0) / recent.length;
      const momentumPct = ((btc - shortMA) / shortMA) * 100;

      // Prediksi ML menggunakan bobot model hasil training Spark/HDFS + momentum mikro
      const basePred = MODEL.intercept
        + (MODEL.coef_btc * btc)
        + (MODEL.coef_eth * eth)
        + (MODEL.coef_sol * sol)
        + (MODEL.coef_btc_chg * btcChg)
        + (MODEL.coef_eth_chg * ethChg)
        + (MODEL.coef_sol_chg * solChg);

      const livePred = basePred * (1 + (momentumPct * 0.6 / 100));
      const diffPred = livePred - btc;
      const diffPct = (diffPred / btc) * 100;

      // Update KPI Cards
      document.getElementById('lastTime').innerText = timeStr + ' WIB';
      document.getElementById('btcPrice').innerText = formatUSD(btc);

      const btcDeltaEl = document.getElementById('btcDelta');
      btcDeltaEl.innerText = (btcChg >= 0 ? '↑ +' : '↓ ') + btcChg.toFixed(2) + '% (24j)';
      btcDeltaEl.className = 'kpi-delta ' + (btcChg >= 0 ? 'up' : 'down');

      document.getElementById('predPrice').innerText = formatUSD(livePred);
      const predDeltaEl = document.getElementById('predDelta');
      predDeltaEl.innerText = (diffPred >= 0 ? '↑ +' : '↓ ') + diffPct.toFixed(2) + '% (' + (diffPred >= 0 ? '+' : '') + formatUSD(diffPred) + ')';
      predDeltaEl.className = 'kpi-delta ' + (diffPred >= 0 ? 'up' : 'down');

      const sigEl = document.getElementById('mlSignal');
      const momEl = document.getElementById('mlMomentum');
      if (momentumPct > 0.002) {{
        sigEl.innerText = '📈 BULLISH';
        momEl.className = 'kpi-delta up';
      }} else if (momentumPct < -0.002) {{
        sigEl.innerText = '📉 BEARISH';
        momEl.className = 'kpi-delta down';
      }} else {{
        sigEl.innerText = '⚖️ SIDEWAYS';
        momEl.className = 'kpi-delta neutral';
      }}
      momEl.innerText = 'Momentum: ' + (momentumPct >= 0 ? '+' : '') + momentumPct.toFixed(4) + '%';

      const drivers = [
        {{ name: `BTC (${{btcChg >= 0 ? '+' : ''}}${{btcChg.toFixed(2)}}%)`, val: Math.abs(btcChg) }},
        {{ name: `ETH (${{ethChg >= 0 ? '+' : ''}}${{ethChg.toFixed(2)}}%)`, val: Math.abs(ethChg) }},
        {{ name: `SOL (${{solChg >= 0 ? '+' : ''}}${{solChg.toFixed(2)}}%)`, val: Math.abs(solChg) }}
      ];
      drivers.sort((a, b) => b.val - a.val);
      document.getElementById('activeDriver').innerText = drivers[0].name;

      // Sliding window: setelah 15 titik, titik waktu paling kiri otomatis hilang/bergeser keluar
      Plotly.extendTraces('liveChart', {{
        x: [[timeStr], [timeStr]],
        y: [[btc], [livePred]]
      }}, [0, 1], MAX_POINTS);
    }}

    updateStream();
    setInterval(updateStream, 2000);
  </script>
</body>
</html>
"""

components.html(realtime_widget_html, height=530)

st.divider()

# 4. GRAFIK BUSINESS INTELLIGENCE & HISTORIS 1 TAHUN (DARI HADOOP HDFS)
st.subheader("📊 Analisis Business Intelligence & Evaluasi Historis (1 Tahun Data HDFS)")
col_bi, col_hist = st.columns([1, 1])

with col_bi:
    fig_bi = px.bar(
        bi_df.sort_values("skor_pengaruh_persen"),
        x="skor_pengaruh_persen", y="faktor_lampau",
        orientation="h", color="korelasi_pearson",
        color_continuous_scale="RdYlGn",
        labels={"skor_pengaruh_persen": "Kontribusi Pengaruh (%)", "faktor_lampau": "Faktor Historis (Lag)"},
        title="1. Business Intelligence: Faktor Hari Lampau yang Memengaruhi Hari Ini"
    )
    fig_bi.update_layout(height=420, margin=dict(t=50, b=30, l=10, r=10))
    st.plotly_chart(fig_bi, use_container_width=True, key="bi_chart")

with col_hist:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=df["date"], y=df["btc_close"], name="Harga Aktual Historis (USD)", line=dict(color="#1f77b4", width=2)))
    fig_hist.add_trace(go.Scatter(x=df["date"], y=df["predicted_next_close"], name="Prediksi Model GBR (USD)", line=dict(color="#ff7f0e", width=2, dash="dot")))
    fig_hist.update_layout(
        title="2. Evaluasi Prediksi 1 Tahun (Aktual vs Prediksi Model)",
        xaxis_title="Tanggal", yaxis_title="Harga BTC (USD)",
        height=420, margin=dict(t=50, b=30, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_hist, use_container_width=True, key="hist_chart")

# 5. TABEL DETEKSI ANOMALI VOLUME PASAR
st.subheader("🚨 3. Log Deteksi Anomali Volume Transaksi Pasar (Z-Score > 2.5)")
anomalies = df[df["is_anomaly"] == True][["date", "btc_close", "btc_volume", "volume_zscore", "eth_close", "sol_close"]].copy()
anomalies.columns = ["Tanggal", "Harga BTC ($)", "Volume BTC", "Skor Anomali (Z-Score)", "Harga ETH ($)", "Harga SOL ($)"]
st.dataframe(anomalies.tail(10), use_container_width=True)