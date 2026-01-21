import React, { useEffect, useRef } from 'react';
import clsx from 'clsx';
import { Terminal } from 'lucide-react';

const LiveLog = ({ logs = [] }) => {
    const endRef = useRef(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    return (
        <div className="col-span-1 md:col-span-4 lg:col-span-3 bg-holon-card rounded-xl border border-slate-700/50 flex flex-col shadow-xl overflow-hidden h-[300px] md:h-auto">
            <div className="bg-slate-900/50 p-3 border-b border-slate-700/50 flex justify-between items-center">
                <div className="flex items-center gap-2 text-holon-accent">
                    <Terminal size={16} />
                    <span className="font-orbitron text-xs tracking-wider">SYSTEM LOGS</span>
                </div>
                <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-holon-dim">LIVE</span>
            </div>

            <div className="flex-1 overflow-y-auto p-3 font-mono text-[11px] space-y-1 log-scrollbar bg-slate-950/30">
                {logs.length === 0 && <div className="text-center text-holon-dim py-10">Waiting for logs...</div>}

                {logs.map((log, i) => {
                    const txt = log.msg || '';
                    const isErr = txt.includes('ERROR') || txt.includes('FATAL');
                    const isWarn = txt.includes('WARNING') || txt.includes('ALERT');
                    const isSuccess = txt.includes('SUCCESS') || txt.includes('PROFIT');

                    return (
                        <div key={i} className={clsx("flex gap-2 items-start opacity-90 hover:opacity-100 transition-opacity", {
                            'text-red-400 font-bold': isErr,
                            'text-yellow-400': isWarn,
                            'text-emerald-400': isSuccess,
                            'text-slate-400': !isErr && !isWarn && !isSuccess
                        })}>
                            <span className="text-slate-600 shrink-0 select-none">[{log.time}]</span>
                            <span className="break-words leading-tight">{txt}</span>
                        </div>
                    );
                })}
                <div ref={endRef} />
            </div>
        </div>
    );
};

export default LiveLog;
