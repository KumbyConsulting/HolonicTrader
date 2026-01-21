/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'holon-bg': '#0f172a',    // Slate 900
                'holon-card': '#1e293b',  // Slate 800
                'holon-accent': '#10b981', // Emerald 500
                'holon-text': '#e2e8f0',   // Slate 200
                'holon-dim': '#64748b',    // Slate 500
            },
            fontFamily: {
                'mono': ['"JetBrains Mono"', 'monospace'],
                'sans': ['Inter', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
