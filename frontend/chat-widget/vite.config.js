import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: 'dist',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    },
    rollupOptions: {
      output: {
        // 輸出單一 JS 檔案，方便 iframe 載入
        entryFileNames: 'chat.js',
        chunkFileNames: 'chat-[hash].js',
        assetFileNames: 'chat.[ext]',
        // 手動控制打包，避免分割
        manualChunks: undefined
      }
    }
  },
  server: {
    port: 5180,
    cors: true,
    // 開發模式允許跨域
    headers: {
      'Access-Control-Allow-Origin': '*'
    },
    // Proxy API 請求到後端（Docker 內使用 app-dev，本機使用 localhost）
    proxy: {
      '/api': {
        target: process.env.API_TARGET || 'http://localhost:8000',
        changeOrigin: true,
        secure: false
      }
    },
    // HMR 設定
    watch: {
      usePolling: true // Docker volume 需要 polling
    }
  },
  // 關閉 SPA history fallback，讓靜態檔案正常返回
  appType: 'mpa'
});
