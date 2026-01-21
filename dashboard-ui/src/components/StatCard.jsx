import React from 'react';
import clsx from 'clsx';

const StatCard = ({ label, value, subValue, icon: Icon, color, className }) => {
    return (
        <div className={clsx("bg-holon-card p-4 rounded-xl border border-slate-700/50 flex flex-col justify-between relative overflow-hidden group hover:border-slate-600 transition-colors", className)}>
            <div className={`absolute right-[-10px] top-[-10px] opacity-10 group-hover:opacity-20 transition-opacity ${color}`}>
                {Icon && <Icon size={64} />}
            </div>

            <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-mono text-holon-dim uppercase tracking-wider">{label}</span>
                {Icon && <Icon size={16} className={clsx("opacity-50", color)} />}
            </div>

            <div className="flex items-baseline gap-2">
                <span className={clsx("text-xl md:text-2xl font-bold font-orbitron", color)}>
                    {value}
                </span>
                {subValue && (
                    <span className="text-xs font-mono text-holon-dim">
                        {subValue}
                    </span>
                )}
            </div>
        </div>
    );
};

export default StatCard;
