import { defineConfig } from 'vite';

// `python server.py` (in Tink-backend-v1) serves the API on :8000. In development, Vite forwards /api to it.
const backend = 'http://127.0.0.1:8000';

export default defineConfig({
  esbuild: { jsx: 'automatic' },
  server: { proxy: { '/api': { target: backend, changeOrigin: false } } },
  preview: { proxy: { '/api': { target: backend, changeOrigin: false } } },
});
