import React from 'react';
import { Radar, Target } from 'lucide-react';
import clsx from 'clsx';

const RadarPanel = ({ items = [] }) => {
    return (
        <div className="bg-holon-card rounded-xl border border-slate-700/50 flex flex-col shadow-lg overflow-hidden h-full">
            <div className="bg-slate-900/50 p-3 border-b border-slate-700/50 flex justify-between items-center">
                <div className="flex items-center gap-2 text-blue-400">
                    <Radar size={16} />
                    <span className="font-orbitron text-xs tracking-wider">MARKET RADAR</span>
                </div>
                <div className="flex gap-2">
                    <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-holon-dim">SCOUT ACTIVE</span>
                </div>
            </div>

            <div className="flex-1 overflow-auto p-0">
                <table className="w-full text-left text-sm">
                    <thead className="bg-slate-950/50 text-[10px] uppercase text-holon-dim font-mono sticky top-0">
                        <tr>
                            <th className="p-2 pl-3">Asset</th>
                            <th className="p-2 text-right">Score</th>
                            <th className="p-2 text-center">Status</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                        {items.length === 0 ? (
                            <tr>
                                <td colSpan="3" className="p-8 text-center text-holon-dim italic text-xs">
                                    Scanning sector alpha...
                                </td>
                            </tr>
                        ) : (
                            items.slice(0, 8).map((item, idx) => (
                                <tr key={idx} className="hover:bg-white/5 transition-colors font-mono text-xs">
                                    <td className="p-2 pl-3 font-bold text-white flex items-center gap-2">
                                        <Target size={12} className="text-blue-500/50" />
                                        {item.symbol}
                                    </td>
                                    <td className="p-2 text-right text-blue-300">
                                        {item.score?.toFixed(1) || '0.0'}
                                    </td>
                                    <td className="p-2 text-center">
                                        <span className={clsx("px-1.5 py-0.5 rounded text-[9px] font-bold uppercase", {
                                            'bg-blue-500/20 text-blue-400': item.status === 'PENDING',
                                            'bg-red-500/20 text-red-400': item.status === 'REJECTED',
                                            'bg-emerald-500/20 text-emerald-400': item.status === 'ACQUIRED'
                                        })}>
                                            {item.status || 'SCAN'}
                                        </span>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default RadarPanel;
