import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const backendHost = process.env.BACKEND_HOST || 'localhost'
const httpTarget = `http://${backendHost}:8000`
const wsTarget = `ws://${backendHost}:8000`

const httpProxy = { target: httpTarget, changeOrigin: true, secure: false }

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
      '/udf': httpProxy,
      '/predictions': httpProxy,
      '/predict': httpProxy,
      '/scripts': httpProxy,
      '/portfolio': httpProxy,
      '/data': httpProxy,
      '/health': httpProxy,
      '/trading': httpProxy,
      '/backtest': httpProxy,
      '/jobs': httpProxy,
      '/api': httpProxy,
      '/ws': {
        target: wsTarget,
        ws: true,
        changeOrigin: true
      }
    }
  }
})
