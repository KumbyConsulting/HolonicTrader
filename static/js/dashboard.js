// Dashboard Logic - Extracted from dashboard.html
const API_URL = '/api';

// --- Utilities ---
function formatMoney(num) {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(num);
}
function formatPct(num) {
    return (num * 100).toFixed(2) + '%';
}

// --- Chart.js Init ---
const ctx = document.getElementById('equityChart').getContext('2d');
const equityChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Equity ($)',
            data: [],
            borderColor: '#10b981', // Accent Green
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 0
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { display: false }, ticks: { color: '#64748b' } },
            y: { grid: { color: '#2d3748' }, ticks: { color: '#64748b' } }
        },
        animation: false // Disable animation for performance
    }
});

// --- Core Loop ---
async function fetchState() {
    try {
        const res = await fetch(`${API_URL}/data`);
        const state = await res.json();
        render(state);
    } catch (e) {
        console.error("Connection Error", e);
    }
}

function render(state) {
    // 1. Status Bar
    const statusBadge = document.getElementById('sys-status-badge');
    statusBadge.innerText = state.status;
    statusBadge.className = `badge ${state.status === 'RUNNING' ? 'bg-success' : 'bg-danger'}`;

    document.getElementById('sys-regime').innerText = state.regime;
    document.getElementById('sys-equity').innerText = formatMoney(state.equity);
    document.getElementById('sys-health').innerText = state.health.toFixed(2);
    document.getElementById('sys-winrate').innerText = state.win_rate ? state.win_rate : "0%";

    // 2. Positions
    const posBody = document.getElementById('position-table-body');
    if (state.positions.length > 0) {
        posBody.innerHTML = state.positions.map(p => {
            const pnlClass = p.pnl_pct >= 0 ? 'text-success' : 'text-danger';
            const side = p.qty > 0 ? 'LONG' : 'SHORT';
            const sideClass = p.qty > 0 ? 'badge bg-success' : 'badge bg-danger';
            return `
                <tr>
                    <td class="fw-bold text-white">${p.symbol}</td>
                    <td><span class="${sideClass}">${side}</span></td>
                    <td class="mono-font">${Math.abs(p.qty).toFixed(4)}</td>
                    <td class="mono-font text-dim">$${p.entry.toFixed(4)}</td>
                    <td class="mono-font">$${p.price.toFixed(4)}</td>
                    <td class="fw-bold ${pnlClass}">${p.pnl_pct.toFixed(2)}%</td>
                    <td class="mono-font">$${Math.abs(p.value).toFixed(2)}</td>
                </tr>
             `;
        }).join('');
    } else {
        posBody.innerHTML = '<tr><td colspan="7" class="text-center text-dim py-4">No active positions.</td></tr>';
    }

    // 3. Radar
    const radarBody = document.getElementById('radar-table-body');
    if (state.radar && state.radar.length > 0) {
        radarBody.innerHTML = state.radar.map((r, i) => `
            <tr>
                <td class="text-dim">#${i + 1}</td>
                <td class="fw-bold">${r.symbol || r.asset || '-'}</td>
                <td class="text-info">${r.score ? r.score.toFixed(2) : '-'}</td>
                <td class="text-dim">${r.reason || '-'}</td>
                <td><span class="badge bg-secondary">WATCH</span></td>
            </tr>
        `).join('');
    }

    // 4. Logs
    const logWindow = document.getElementById('log-window');
    // Only update if new logs? Simplistic: Re-render last 50
    if (state.logs.length > 0) {
        logWindow.innerHTML = state.logs.slice().reverse().map(l => {
            let cls = '';
            const txt = l.msg || ''; // Safe Guard
            if (txt.includes('ERROR') || txt.includes('FATAL')) cls = 'error';
            else if (txt.includes('WARNING') || txt.includes('ALERT')) cls = 'warn';
            else if (txt.includes('SUCCESS') || txt.includes('PROFIT')) cls = 'success';

            return `<div class="log-entry ${cls}"><span class="text-dim">[${l.time}]</span> ${txt}</div>`;
        }).join('');
    }

    // 5. Chart Update
    if (state.equity_history && state.equity_history.length > 0) {
        const labels = state.equity_history.map(p => p.t);
        const data = state.equity_history.map(p => p.y);

        // Only update if changed
        if (equityChart.data.labels.length !== labels.length || equityChart.data.datasets[0].data[equityChart.data.datasets[0].data.length - 1] !== data[data.length - 1]) {
            equityChart.data.labels = labels;
            equityChart.data.datasets[0].data = data;
            equityChart.update();
        }
    }
}

// --- Commands ---
async function sendCmd(cmd) {
    const payload = { command: cmd, data: {} };

    if (cmd === 'start') {
        payload.data = {
            symbol: document.getElementById('cfg-sym').value,
            allocation: document.getElementById('cfg-alloc').value,
            leverage: document.getElementById('cfg-lev').value,
            micro: true
        };

        // Button Feedback
        document.getElementById('btn-start').innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i>STARTING...';
    }

    try {
        const res = await fetch(`${API_URL}/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const json = await res.json();

        // Reset Button
        if (cmd === 'start') document.getElementById('btn-start').innerHTML = '<i class="fa-solid fa-play me-2"></i>INITIATE';

    } catch (e) {
        alert("Command Failed: " + e);
    }

    // Force immediate update
    fetchState();
}

// Init
// setInterval(fetchState, 1000); // Polling Removed
// fetchState(); // Initial Load

// --- WebSocket Init ---
const socket = io();

socket.on('connect', () => {
    console.log(">> Connected to Real-time Stream");
    document.getElementById('sys-status-badge').classList.add('pulse'); // Visual feedback
    fetchState(); // Load initial state (REST fallback)
});

socket.on('state_update', (data) => {
    // Real-time Push
    render(data);
});

socket.on('disconnect', () => {
    console.log(">> Disconnected");
    const badge = document.getElementById('sys-status-badge');
    badge.innerText = 'OFFLINE';
    badge.className = 'badge bg-secondary';
});
