/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                bg: '#fffff0', // Ivory
                paper: '#f5f5e6', // Slightly darker ivory for surface
                charcoal: '#1a1a1a', // Primary Text
                stone: '#666666', // Secondary Text
                accent: '#e0501a', // Burnt Orange
            },
            fontFamily: {
                serif: ['Italiana', 'serif'],
                sans: ['Manrope', 'sans-serif'],
            },
        },
    },
    plugins: [],
}
