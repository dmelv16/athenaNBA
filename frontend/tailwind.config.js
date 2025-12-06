// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        gray: {
          750: '#2d3748',
          850: '#1a202c',
        }
      }
    },
  },
  plugins: [],
}