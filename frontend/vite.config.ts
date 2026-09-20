import { defineConfig } from 'vite';

export default defineConfig({
  base: '/app/',
  server: { proxy: { '/api': 'http://127.0.0.1:8400' } },
});
