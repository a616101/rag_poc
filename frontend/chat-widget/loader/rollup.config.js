import terser from '@rollup/plugin-terser';

export default {
  input: 'loader/src/widget-loader.js',
  output: [
    // 生產用 (dist)
    {
      file: 'dist/widget.js',
      format: 'iife',
      sourcemap: false
    },
    // 開發用 (public - Vite dev server 會自動提供)
    {
      file: 'public/widget.js',
      format: 'iife',
      sourcemap: false
    }
  ],
  plugins: [
    terser({
      compress: {
        drop_console: true,
        drop_debugger: true
      },
      mangle: true,
      format: {
        comments: false
      }
    })
  ]
};
