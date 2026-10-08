import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/demo': {
        target: 'http://localhost:3003',
        changeOrigin: true,
      },
    },
  },
})
