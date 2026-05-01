module.exports = {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{ts,tsx,js,jsx}'
  ],
  theme: {
    container: {
      center: true,
      padding: '1rem',
      screens: {
        '2xl': '1400px',
      },
    },
    extend: {
      fontFamily: {
        sans: ['Trebuchet MS', 'Lucida Grande', 'Lucida Sans Unicode', 'Lucida Sans', 'Tahoma', 'sans-serif'],
        mono: ['Monaco', 'Courier New', 'monospace'],
      },
      colors: {
        // TradingView color palette
        tv: {
          // Theme-aware colors (adapt to light/dark via CSS variables)
          'bg-primary':      'hsl(var(--tv-bg-primary) / <alpha-value>)',
          'bg-secondary':    'hsl(var(--tv-bg-secondary) / <alpha-value>)',
          'bg-tertiary':     'hsl(var(--tv-bg-tertiary) / <alpha-value>)',
          'bg-elevated':     'hsl(var(--tv-bg-elevated) / <alpha-value>)',
          'bg-hover':        'hsl(var(--tv-bg-hover) / <alpha-value>)',
          'border-primary':  'hsl(var(--tv-border-primary) / <alpha-value>)',
          'border-secondary':'hsl(var(--tv-border-secondary) / <alpha-value>)',
          'border-hover':    'hsl(var(--tv-border-hover) / <alpha-value>)',
          'text-primary':    'hsl(var(--tv-text-primary) / <alpha-value>)',
          'text-secondary':  'hsl(var(--tv-text-secondary) / <alpha-value>)',
          'text-tertiary':   'hsl(var(--tv-text-tertiary) / <alpha-value>)',
          'text-disabled':   'hsl(var(--tv-text-disabled) / <alpha-value>)',

          // Brand/semantic colors — same in both themes
          'blue':        '#2962FF',
          'blue-hover':  '#1E53E5',
          'blue-pressed':'#1948CC',
          'green':       '#089981',
          'green-hover': '#06876E',
          'red':         '#F23645',
          'red-hover':   '#D9303E',
          'orange':      '#FF9800',
          'yellow':      '#FFC107',
        },
        
        // Shadcn-compatible color system
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        success: {
          DEFAULT: '#089981',
          foreground: '#FFFFFF',
        },
        warning: {
          DEFAULT: '#FF9800',
          foreground: '#FFFFFF',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
        'fade-in': {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        'slide-in-from-top': {
          from: { transform: 'translateY(-10px)', opacity: '0' },
          to: { transform: 'translateY(0)', opacity: '1' },
        },
        'slide-in-from-left': {
          from: { transform: 'translateX(-10px)', opacity: '0' },
          to: { transform: 'translateX(0)', opacity: '1' },
        },
        'pulse-subtle': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.8' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        'fade-in': 'fade-in 0.2s ease-out',
        'slide-in': 'slide-in-from-top 0.2s ease-out',
        'slide-in-left': 'slide-in-from-left 0.2s ease-out',
        'pulse-subtle': 'pulse-subtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 2s linear infinite',
      },
      boxShadow: {
        'tv-sm': '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
        'tv-md': '0 2px 4px 0 rgba(0, 0, 0, 0.4)',
        'tv-lg': '0 4px 8px 0 rgba(0, 0, 0, 0.5)',
        'tv-xl': '0 8px 16px 0 rgba(0, 0, 0, 0.6)',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
}
