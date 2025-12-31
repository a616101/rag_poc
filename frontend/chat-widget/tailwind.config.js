/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{svelte,js,ts}'
  ],
  theme: {
    extend: {
      colors: {
        // 使用 CSS 變數讓主題色可動態調整
        primary: {
          DEFAULT: 'var(--widget-primary, #6366f1)',
          hover: 'var(--widget-primary-hover, #4f46e5)',
          light: 'var(--widget-primary-light, #e0e7ff)'
        }
      },
      animation: {
        'bounce-dot': 'bounce-dot 1.4s infinite ease-in-out'
      },
      keyframes: {
        'bounce-dot': {
          '0%, 80%, 100%': { transform: 'translateY(0)' },
          '40%': { transform: 'translateY(-6px)' }
        }
      }
    }
  },
  plugins: []
};
