import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  server: {
    port: 5173,
    open: true,
    proxy: {
      '/udf': {
        target: 'http://backend:8000',
        changeOrigin: true,
        secure: false
      },
      '/predictions': {
        target: 'http://backend:8000',
        changeOrigin: true,
        secure: false
      },
      '/predict': {
        target: 'http://backend:8000',
        changeOrigin: true,
        secure: false
      },
      '/scripts': {
        target: 'http://backend:8000',
        changeOrigin: true,
        secure: false
      },
      '/portfolio': {
        target: 'http://backend:8000',
        changeOrigin: true,
        secure: false
      },
      '/health': {
        target: 'http://backend:8000',
        changeOrigin: true,
        secure: false
      },
      '/trading': {
        target: 'http://backend:8000',
        changeOrigin: true,
        secure: false
      },
      '/backtest': {
        target: 'http://backend:8000',
        changeOrigin: true,
        secure: false
      },
      '/jobs': {
        target: 'http://backend:8000',
        changeOrigin: true,
        secure: false
      },
      '/ws': {
        target: 'ws://backend:8000',
        ws: true,
        changeOrigin: true
      },
      '/api': {
        target: 'http://backend:8000',
        changeOrigin: true,
        secure: false
      }
    }
  }
})
