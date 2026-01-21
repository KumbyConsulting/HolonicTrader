import React, { useState } from 'react';
import { SocketProvider, useSocket } from './context/SocketContext';
import DashboardLayout from './components/DashboardLayout';
import LiveLog from './components/LiveLog';
import StatCard from './components/StatCard';
import EquityChart from './components/EquityChart';
import RadarPanel from './components/RadarPanel';
import { Activity, Shield, Wallet, Play, Square, AlertTriangle, TrendingUp, DollarSign } from 'lucide-react';
import clsx from 'clsx';

function Dashboard() {
  const { isConnected, systemState, sendCommand } = useSocket();
  const [loadingCmd, setLoadingCmd] = useState(null);

  // Initial Empty State
  const state = systemState || {
    status: 'DISCONNECTED',
    health: 0,
    equity: 0,
    balance: 0,
    pnl: 0,
    regime: '---',
    logs: [],
    positions: [],
    radar: [],
    equity_history: []
  };

  const handleCmd = async (cmd) => {
    setLoadingCmd(cmd);
    await sendCommand(cmd, {
      symbol: 'BTC/USDT',
      leverage: 5.0,
      allocation: 0.1
    });
    setTimeout(() => setLoadingCmd(null), 1000); // UI Cooloff
  };

  const isRunning = state.status === 'RUNNING';

  return (
    <DashboardLayout>
      {/* --- COLUMN 1: FINANCIALS & CONTROLS (3 Cols) --- */}
      <div className="col-span-1 md:col-span-12 lg:col-span-3 flex flex-col gap-4">
        {/* Status KPI Group */}
        <div className="grid grid-cols-2 lg:grid-cols-1 gap-4">
          <StatusBadge status={state.status} health={state.health} />
          <StatCard
            label="TOTAL EQUITY"
            value={`$${state.equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
            icon={Wallet}
            color="text-emerald-400"
          />
          <StatCard
            label="SESSION PNL"
            value={`${state.pnl >= 0 ? '+' : ''}$${state.pnl.toFixed(2)}`}
            icon={TrendingUp}
            color={state.pnl >= 0 ? "text-emerald-400" : "text-red-400"}
          />
        </div>

        {/* Controls */}
        <div className="bg-holon-card p-4 rounded-xl border border-slate-700/50 flex flex-col gap-3 mt-auto">
          <p className="text-[10px] uppercase text-holon-dim font-mono tracking-wider mb-1">Command Override</p>
          <ActionButton
            label={isRunning ? "SYSTEM ACTIVE" : "INITIATE SEQUENCE"}
            icon={Play}
            onClick={() => handleCmd('start')}
            active={isRunning}
            loading={loadingCmd === 'start'}
            disabled={isRunning}
            color="emerald"
            fullWidth
          />
          <ActionButton
            label="HALT OPERATIONS"
            icon={Square}
            onClick={() => handleCmd('stop')}
            loading={loadingCmd === 'stop'}
            disabled={state.status === 'STOPPED'}
            color="red"
            fullWidth
          />
          <button
            onClick={() => handleCmd('panic')}
            className="mt-2 text-[10px] text-red-500/40 hover:text-red-500 transition-colors flex items-center justify-center gap-2 font-mono uppercase bg-red-950/20 py-2 rounded border border-red-900/30 hover:border-red-500/50"
          >
            <AlertTriangle size={12} />
            Emergency Liquidation Protocol
          </button>
        </div>
      </div>

      {/* --- COLUMN 2: VISUALIZATION (6 Cols) --- */}
      <div className="col-span-1 md:col-span-12 lg:col-span-6 flex flex-col gap-4">
        {/* Equity Chart */}
        <div className="h-64 lg:h-80 bg-holon-card rounded-xl border border-slate-700/50 p-4 relative flex flex-col">
          <div className="flex justify-between items-start mb-2">
            <div>
              <p className="text-[10px] uppercase text-holon-dim font-mono tracking-wider">Performance Vector</p>
              <h3 className="text-lg font-bold text-white font-orbitron">EQUITY CURVE</h3>
            </div>
            <div className="text-right">
              <p className="text-xs text-slate-500 font-mono">LIVE FEED</p>
            </div>
          </div>
          <div className="flex-1 w-full min-h-0">
            <EquityChart dataPoints={state.equity_history} />
          </div>
        </div>

        {/* Radar Panel */}
        <div className="flex-1 min-h-[200px]">
          <RadarPanel items={state.radar} />
        </div>
      </div>

      {/* --- COLUMN 3: POSITIONS & LOGS (3 Cols) --- */}
      <div className="col-span-1 md:col-span-12 lg:col-span-3 flex flex-col gap-4 h-full">
        {/* Regime Indicator */}
        <div className="bg-gradient-to-r from-blue-900/40 to-slate-900/40 p-3 rounded-xl border border-blue-800/30 flex items-center justify-between">
          <span className="text-[10px] font-mono text-blue-300 uppercase">Market Regime</span>
          <span className="font-bold font-orbitron text-sm text-blue-100">{state.regime}</span>
        </div>

        {/* Positions Table (Compact) */}
        <div className="flex-1 bg-holon-card rounded-xl border border-slate-700/50 flex flex-col overflow-hidden min-h-[300px]">
          <div className="bg-slate-900/50 p-3 border-b border-slate-700/50 flex justify-between">
            <span className="font-orbitron text-xs tracking-wider text-holon-accent">ACTIVE POSITIONS</span>
            <span className="text-xs text-holon-dim">{state.positions.length} OPEN</span>
          </div>
          <div className="flex-1 overflow-auto p-0">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-950/50 text-[10px] uppercase text-holon-dim font-mono sticky top-0">
                <tr>
                  <th className="p-2 pl-3">Sym</th>
                  <th className="p-2 text-right">Size</th>
                  <th className="p-2 text-right">PnL%</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {state.positions.length === 0 ? (
                  <tr>
                    <td colSpan="3" className="p-8 text-center text-holon-dim italic text-xs">
                      No active vectors.
                    </td>
                  </tr>
                ) : (
                  state.positions.map((p, idx) => (
                    <tr key={idx} className="hover:bg-white/5 transition-colors font-mono text-xs">
                      <td className="p-2 pl-3 font-bold text-white">
                        <div className="flex flex-col">
                          <span>{p.symbol}</span>
                          <span className={clsx("text-[9px]", p.qty > 0 ? "text-emerald-500" : "text-red-500")}>
                            {p.qty > 0 ? 'LONG' : 'SHORT'}
                          </span>
                        </div>
                      </td>
                      <td className="p-2 text-right text-slate-300">{Math.abs(p.qty).toFixed(3)}</td>
                      <td className={clsx("p-2 text-right font-bold", p.pnl_pct >= 0 ? "text-emerald-400" : "text-red-400")}>
                        {p.pnl_pct > 0 ? '+' : ''}{p.pnl_pct.toFixed(2)}%
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Logs Area */}
        <LiveLog logs={state.logs} />
      </div>
    </DashboardLayout>
  );
}

// Helper Components
const StatusBadge = ({ status, health }) => {
  const isRunning = status === 'RUNNING';
  return (
    <div className="bg-holon-card p-4 rounded-xl border border-slate-700/50 flex items-center gap-4 relative overflow-hidden">
      <div className={clsx("h-2 w-2 rounded-full absolute top-3 right-3 animate-pulse", isRunning ? "bg-emerald-500" : "bg-red-500")} />
      <div className={clsx("p-3 rounded-lg", isRunning ? "bg-emerald-500/10 text-emerald-400" : "bg-red-500/10 text-red-500")}>
        <Activity size={24} />
      </div>
      <div>
        <p className="text-[10px] font-mono text-holon-dim uppercase tracking-wider">SYSTEM STATUS</p>
        <p className={clsx("text-lg font-bold font-orbitron", isRunning ? "text-emerald-400" : "text-red-400")}>
          {status}
        </p>
        <div className="flex items-center gap-2 mt-1">
          <div className="h-1 w-12 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full bg-purple-500 transition-all duration-500" style={{ width: `${health}%` }} />
          </div>
          <span className="text-[9px] text-purple-400 font-mono">{health.toFixed(0)}% HP</span>
        </div>
      </div>
    </div>
  );
};

const ActionButton = ({ label, icon: Icon, onClick, loading, disabled, color, active, fullWidth }) => {
  const baseClass = "flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-bold font-orbitron tracking-wider text-xs transition-all shadow-lg active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed";
  const colors = {
    emerald: "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/20",
    red: "bg-red-600 hover:bg-red-500 text-white shadow-red-900/20"
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className={clsx(baseClass, colors[color], fullWidth && "w-full")}
    >
      {loading ? <span className="animate-spin">⏳</span> : <Icon size={14} fill={active ? "currentColor" : "none"} />}
      {label}
    </button>
  );
};

export default function App() {
  return (
    <SocketProvider>
      <Dashboard />
    </SocketProvider>
  );
}
