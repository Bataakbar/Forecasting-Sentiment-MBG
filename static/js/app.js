// Pustaka Frontend Logika - Dashboard Forecasting Sentimen MBG

let dashboardData = null;
let trendChartPosInstance = null;
let trendChartNegInstance = null;
let donutChartInstance = null;
let currentChartMode = 'raw'; // 'raw' atau 'smooth'

document.addEventListener('DOMContentLoaded', () => {
  fetchDashboardData();
});

// 1. Ambil data utama dari API dashboard Flask
async function fetchDashboardData() {
  try {
    const response = await fetch('/api/dashboard');
    if (!response.ok) throw new Error('Gagal mengambil data dashboard');
    
    dashboardData = await response.json();
    
    // Update teks judul & deskripsi secara dinamis berdasarkan panjang data
    document.getElementById('trendChartPosTitle').innerText = `Tren & Forecast Sentimen Positif (${dashboardData.history.dates_display.length} Hari Terakhir + Forecast ${dashboardData.forecast.pos.length} Hari)`;
    document.getElementById('trendChartNegTitle').innerText = `Tren & Forecast Sentimen Negatif (${dashboardData.history.dates_display.length} Hari Terakhir + Forecast ${dashboardData.forecast.neg.length} Hari)`;
    document.getElementById('forecastInfoText').innerText = `Proyeksi ${dashboardData.forecast.pos.length} hari ke depan dihitung secara autoregresif menggunakan data historis 14 hari terakhir melalui model CNN-LSTM terpisah untuk tren positif dan negatif.`;
    document.getElementById('forecastTableTitle').innerText = `Forecast Harian (${dashboardData.forecast.pos.length} Hari Ke Depan)`;
    
    // Update antarmuka secara dinamis
    updateKPIs(dashboardData.metrics);
    renderTrendChart(dashboardData.history, dashboardData.forecast);
    renderDonutChart(dashboardData.donut);
    renderTopics(dashboardData.topics);
    renderForecastTable(dashboardData.forecast, dashboardData.history);
    renderComments(dashboardData.recent_comments);
    initSimulatorInputs(dashboardData.history);
    
  } catch (error) {
    console.error('Error:', error);
    const banner = document.getElementById('errorBanner');
    const bannerText = document.getElementById('errorBannerText');
    if (banner && bannerText) {
      bannerText.innerText = 'Gagal memuat data dashboard: ' + error.message + ' (Buka konsol browser [F12] untuk melihat detail teknis)';
      banner.style.display = 'flex';
    } else {
      alert('Terjadi kesalahan saat memuat data: ' + error.message);
    }
  }
}

// 2. Pembaruan KPI Utama di Atas
function updateKPIs(metrics) {
  document.getElementById('totalMention').innerText = metrics.total_mention;
  document.getElementById('latestPos').innerText = metrics.latest_pos;
  document.getElementById('latestNeg').innerText = metrics.latest_neg;
  document.getElementById('latestNeu').innerText = metrics.latest_neu;
  document.getElementById('modelAccuracy').innerText = metrics.model_accuracy;
  document.getElementById('f1Score').innerText = metrics.f1_score;
  
  // Positif Change
  const posContainer = document.getElementById('posChangeContainer');
  const posIcon = document.getElementById('posChangeIcon');
  const posText = document.getElementById('posChangeText');
  posText.innerText = metrics.pos_change.replace('+', '').replace('-', '');
  
  if (metrics.pos_change_up) {
    posContainer.className = 'metric-change up';
    posIcon.className = 'ti ti-trending-up';
  } else {
    posContainer.className = 'metric-change down';
    posIcon.className = 'ti ti-trending-down';
  }
  
  // Negatif Change
  const negContainer = document.getElementById('negChangeContainer');
  const negIcon = document.getElementById('negChangeIcon');
  const negText = document.getElementById('negChangeText');
  negText.innerText = metrics.neg_change.replace('+', '').replace('-', '');
  
  if (metrics.neg_change_up) {
    negContainer.className = 'metric-change up';
    negIcon.className = 'ti ti-trending-up';
  } else {
    negContainer.className = 'metric-change down';
    negIcon.className = 'ti ti-trending-down';
  }
}

// 3. Render Grafik Tren Utama (Pemisahan Positif & Negatif)
function renderTrendChart(history, forecast) {
  const ctxPos = document.getElementById('trendChartPos').getContext('2d');
  const ctxNeg = document.getElementById('trendChartNeg').getContext('2d');
  
  // Tentukan data historis yang akan digambar berdasarkan mode (raw vs smooth)
  const posHist = currentChartMode === 'smooth' ? history.pos_smooth : history.pos_raw;
  const negHist = currentChartMode === 'smooth' ? history.neg_smooth : history.neg_raw;
  
  // Gabungkan label (historis + ramalan)
  const allLabels = [...history.dates_display, ...forecast.dates_display];
  
  // Menggabungkan data historis dan peramalan secara dinamis
  const posActual = [...posHist, ...Array(forecast.pos.length).fill(null)];
  const negActual = [...negHist, ...Array(forecast.neg.length).fill(null)];
  
  const posForecast = [...Array(posHist.length - 1).fill(null), posHist[posHist.length - 1], ...forecast.pos];
  const negForecast = [...Array(negHist.length - 1).fill(null), negHist[negHist.length - 1], ...forecast.neg];
  
  // Hancurkan instance chart lama jika sudah ada
  if (trendChartPosInstance) {
    trendChartPosInstance.destroy();
  }
  if (trendChartNegInstance) {
    trendChartNegInstance.destroy();
  }
  
  // Opsi Skala Chart Standar
  const commonScales = {
    x: {
      grid: { display: false },
      ticks: { maxTicksLimit: 10, font: { size: 10, family: 'DM Sans' } }
    },
    y: {
      grid: { color: 'rgba(0, 0, 0, 0.05)' },
      ticks: { font: { size: 10, family: 'DM Sans' }, callback: value => value + '%' },
      min: 0,
      max: 100
    }
  };

  // Opsi Tooltip dan Legend
  const commonPlugins = {
    legend: { display: false },
    tooltip: {
      mode: 'index',
      intersect: false,
      callbacks: {
        label: function(context) {
          let label = context.dataset.label || '';
          if (label) label += ': ';
          if (context.parsed.y !== null) label += context.parsed.y.toFixed(1) + '%';
          return label;
        }
      }
    }
  };

  // 1. Gambar Grafik Tren Positif
  trendChartPosInstance = new Chart(ctxPos, {
    type: 'line',
    data: {
      labels: allLabels,
      datasets: [
        {
          label: 'Positif (Aktual)',
          data: posActual,
          borderColor: '#3B6D11',
          backgroundColor: 'rgba(59, 109, 17, 0.06)',
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 2
        },
        {
          label: 'Forecast Positif',
          data: posForecast,
          borderColor: '#185FA5',
          borderDash: [5, 4],
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: commonPlugins,
      scales: commonScales
    }
  });

  // 2. Gambar Grafik Tren Negatif
  trendChartNegInstance = new Chart(ctxNeg, {
    type: 'line',
    data: {
      labels: allLabels,
      datasets: [
        {
          label: 'Negatif (Aktual)',
          data: negActual,
          borderColor: '#A32D2D',
          backgroundColor: 'rgba(163, 45, 45, 0.04)',
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 2
        },
        {
          label: 'Forecast Negatif',
          data: negForecast,
          borderColor: '#E24B4A',
          borderDash: [5, 4],
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: commonPlugins,
      scales: commonScales
    }
  });
}

// Kontrol Toggle Filter Tampilan Grafik (Raw vs Moving Average)
function setChartMode(mode) {
  if (currentChartMode === mode) return;
  
  currentChartMode = mode;
  document.getElementById('btnRaw').classList.toggle('active', mode === 'raw');
  document.getElementById('btnSmooth').classList.toggle('active', mode === 'smooth');
  
  if (dashboardData) {
    renderTrendChart(dashboardData.history, dashboardData.forecast);
  }
}

// 4. Render Donut Chart Distribusi Saat Ini
function renderDonutChart(donut) {
  const ctx = document.getElementById('donutChart').getContext('2d');
  
  if (donutChartInstance) {
    donutChartInstance.destroy();
  }
  
  donutChartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: donut.labels,
      datasets: [{
        data: donut.values,
        backgroundColor: ['#639922', '#E24B4A', '#B4B2A9'],
        borderWidth: 0,
        hoverOffset: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '74%',
      plugins: {
        legend: {
          display: true,
          position: 'right',
          labels: {
            boxWidth: 10,
            font: {
              size: 11,
              family: 'DM Sans'
            },
            padding: 12
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return ` ${context.label}: ${context.raw.toFixed(1)}%`;
            }
          }
        }
      }
    }
  });
}

// 5. Render Topik Paling Banyak Dibahas (Bar List)
function renderTopics(topics) {
  const container = document.getElementById('topicListContainer');
  container.innerHTML = '';
  
  if (topics.length === 0) {
    container.innerHTML = '<p style="font-size:12px; color:var(--color-text-secondary); text-align:center;">Tidak ada topik yang diekstrak.</p>';
    return;
  }
  
  topics.forEach(t => {
    const row = document.createElement('div');
    row.className = 'topic-row';
    row.innerHTML = `
      <span class="topic-name">${t.name}</span>
      <div class="topic-bar-wrap">
        <div class="topic-bar-fill" style="width: 0%; background: ${t.color};"></div>
      </div>
      <span class="topic-vol">${t.vol}</span>
    `;
    container.appendChild(row);
    
    // Animasi bar pengisi setelah ditambahkan ke DOM
    setTimeout(() => {
      row.querySelector('.topic-bar-fill').style.width = `${Math.min(100, t.pct)}%`;
    }, 100);
  });
}

// 6. Render Tabel Forecast Harian
function renderForecastTable(forecast, history) {
  const tbody = document.getElementById('forecastTableBody');
  tbody.innerHTML = '';
  
  // Dapatkan nilai terakhir data aktual sebagai pembanding tren pertama
  let prevPos = history.pos_raw[history.pos_raw.length - 1];
  
  forecast.dates_display.forEach((d, idx) => {
    const curPos = forecast.pos[idx];
    const curNeg = forecast.neg[idx];
    
    // Analisis Tren dibanding langkah sebelumnya
    let trendPillClass = 'pill-neu';
    let trendText = '→ Stabil';
    
    const diff = curPos - prevPos;
    if (diff > 0.5) {
      trendPillClass = 'pill-pos';
      trendText = '↑ Naik';
    } else if (diff < -0.5) {
      trendPillClass = 'pill-neg';
      trendText = '↓ Turun';
    }
    
    prevPos = curPos; // Simpan untuk iterasi berikutnya
    
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-weight:500;">${d}</td>
      <td style="color:var(--color-positive); font-weight:600;">${curPos.toFixed(1)}%</td>
      <td style="color:var(--color-negative); font-weight:600;">${curNeg.toFixed(1)}%</td>
      <td><span class="pill ${trendPillClass}">${trendText}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// 7. Render Komentar / Postingan Terbaru
function renderComments(comments) {
  const container = document.getElementById('commentFeedContainer');
  container.innerHTML = '';
  
  if (comments.length === 0) {
    container.innerHTML = '<p style="font-size:12px; color:var(--color-text-secondary); text-align:center;">Tidak ada komentar.</p>';
    return;
  }
  
  comments.forEach(c => {
    const div = document.createElement('div');
    div.className = 'comment-item';
    div.innerHTML = `
      <div class="comment-header">
        <span class="comment-meta">
          <i class="ti ti-brand-twitter"></i> <strong>${c.platform}</strong> · ${c.date}
        </span>
        <span class="pill ${c.sentiment_class}">${c.sentiment}</span>
      </div>
      <p class="comment-body">"${c.text}"</p>
    `;
    container.appendChild(div);
  });
}

// 8. Inisialisasi Bidang Input Simulator Kustom
function initSimulatorInputs(history) {
  const grid = document.getElementById('simulatorInputGrid');
  grid.innerHTML = '';
  
  const posRaw = history.pos_raw;
  const negRaw = history.neg_raw;
  
  // Ambil 14 data terakhir dari data historis
  const startIdx = posRaw.length - 14;
  
  for (let i = 0; i < 14; i++) {
    const idx = startIdx + i;
    const defaultPos = posRaw[idx] ? posRaw[idx].toFixed(1) : '50.0';
    const defaultNeg = negRaw[idx] ? negRaw[idx].toFixed(1) : '25.0';
    
    const dayCol = document.createElement('div');
    dayCol.className = 'sim-day-col';
    dayCol.innerHTML = `
      <div class="sim-day-label">H-${14 - i}</div>
      <div class="sim-input-group">
        <label>Pos %</label>
        <input type="number" step="0.1" min="0" max="100" class="sim-field sim-pos" data-day="${i}" value="${defaultPos}">
      </div>
      <div class="sim-input-group" style="margin-top:2px;">
        <label>Neg %</label>
        <input type="number" step="0.1" min="0" max="100" class="sim-field sim-neg" data-day="${i}" value="${defaultNeg}">
      </div>
    `;
    grid.appendChild(dayCol);
  }
}

// Mengembalikan input simulator ke data aktual rill
function loadDefaultSimulatorData() {
  if (dashboardData) {
    initSimulatorInputs(dashboardData.history);
    document.getElementById('simulatorResult').style.display = 'none';
  }
}

// Mengirimkan data simulator ke API kustom forecast
async function runCustomForecast(event) {
  event.preventDefault();
  
  const posInputs = document.querySelectorAll('.sim-field.sim-pos');
  const negInputs = document.querySelectorAll('.sim-field.sim-neg');
  
  const posHistory = [];
  const negHistory = [];
  
  for (let i = 0; i < 14; i++) {
    posHistory.push(parseFloat(posInputs[i].value));
    negHistory.push(parseFloat(negInputs[i].value));
  }
  
  try {
    const response = await fetch('/api/custom-forecast', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        pos_history: posHistory,
        neg_history: negHistory
      })
    });
    
    if (!response.ok) throw new Error('Gagal menghitung ramalan kustom');
    
    const result = await response.json();
    
    // Tampilkan hasil prediksi kustom
    document.getElementById('simResultPos').innerText = `${result.next_pos.toFixed(1)}%`;
    document.getElementById('simResultNeg').innerText = `${result.next_neg.toFixed(1)}%`;
    document.getElementById('simulatorResult').style.display = 'flex';
    
  } catch (error) {
    console.error('Error:', error);
    alert('Terjadi kesalahan simulasi: ' + error.message);
  }
}
