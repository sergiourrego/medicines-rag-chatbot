/** @type {import('tailwindcss').Config} */
import daisyui from "daisyui"

export default {
  content: [
    "./src/App.jsx",
    "./index.html"
  ],
  theme: {
    extend: {},
  },
  plugins: [
    daisyui,
  ],
}

