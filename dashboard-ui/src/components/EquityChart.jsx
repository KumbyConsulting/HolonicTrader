import React from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Filler,
    Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Filler,
    Legend
);

export const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: false,
        },
        tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: 'rgba(15, 23, 42, 0.9)',
            titleColor: '#e2e8f0',
            bodyColor: '#e2e8f0',
            borderColor: 'rgba(51, 65, 85, 0.5)',
            borderWidth: 1,
        },
    },
    scales: {
        x: {
            display: false, // Minimalist look
            grid: {
                display: false,
            },
        },
        y: {
            display: true,
            position: 'right',
            grid: {
                color: 'rgba(51, 65, 85, 0.1)',
            },
            ticks: {
                color: '#64748b',
                font: {
                    family: 'JetBrains Mono',
                    size: 10,
                },
                callback: function (value) {
                    return '$' + value;
                }
            },
            border: {
                display: false,
            }
        },
    },
    interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
    },
    elements: {
        point: {
            radius: 0,
            hitRadius: 10,
            hoverRadius: 4
        }
    }
};

const EquityChart = ({ dataPoints = [] }) => {
    // dataPoints is array of { t: "HH:MM", y: 1234.56 }

    // Safety check for empty data
    if (!dataPoints || dataPoints.length === 0) {
        return (
            <div className="h-full w-full flex items-center justify-center text-holon-dim text-xs font-mono">
                WAITING FOR DATA...
            </div>
        );
    }

    const labels = dataPoints.map(d => d.t);
    const values = dataPoints.map(d => d.y);

    // Determine color based on trend (Green if last > first)
    const isProfit = values.length > 1 ? values[values.length - 1] >= values[0] : true;
    const color = isProfit ? '#10b981' : '#ef4444'; // Emerald vs Red
    const bgGradient = (context) => {
        const ctx = context.chart.ctx;
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, isProfit ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)');
        gradient.addColorStop(1, 'rgba(15, 23, 42, 0)');
        return gradient;
    };

    const data = {
        labels,
        datasets: [
            {
                fill: true,
                label: 'Equity',
                data: values,
                borderColor: color,
                backgroundColor: bgGradient,
                tension: 0.1, // Slight curve
                borderWidth: 2,
            },
        ],
    };

    return <Line options={options} data={data} />;
};

export default EquityChart;
