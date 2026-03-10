import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    host: true, // same as 0.0.0.0
    port: 3000,
    strictPort: true
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/setupTests.js',
    globals: true,
    coverage: {
      exclude: [
        'build/**',
        'dist/**',
        'node_modules/**',
        'src/index.js',
        'src/main.jsx',
        'src/reportWebVitals.js',
        'eslint.config.js',
        'vite.config.js',
      ]
    }
  }
})
