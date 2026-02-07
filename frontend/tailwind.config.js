/* eslint-disable @typescript-eslint/no-require-imports */
/** @type {import('tailwindcss').Config} */
module.exports = {
    // Performance optimizations
    future: {
        hoverOnlyWhenSupported: true,
    },
    content: {
        files: [
            './app/**/*.{ts,tsx}',
            './components/**/*.{ts,tsx}',
            './lib/**/*.{ts,tsx}',
        ],
        // Relative paths improve file watching performance
        relative: true,
    },
    // Safelist commonly used dynamic classes to avoid rescanning
    safelist: [
        // Grid columns
        { pattern: /grid-cols-(1|2|3|4|5|6|12)/ },
        { pattern: /col-span-(1|2|3|4|6|12)/ },
        // Gaps
        { pattern: /gap-(0|1|2|3|4|5|6|8)/ },
        // Margins and paddings
        { pattern: /[mp][xy]?-(0|1|2|3|4|5|6|8|10|12|16|20)/ },
        // Text sizes
        { pattern: /text-(xs|sm|base|lg|xl|2xl|3xl|4xl)/ },
        // Font weights
        { pattern: /font-(normal|medium|semibold|bold)/ },
        // Colors (primary, secondary, accent - common shades)
        { pattern: /bg-(primary|secondary|accent)-(50|100|500|600|700)/ },
        { pattern: /text-(primary|secondary|accent)-(50|100|500|600|700|800|900)/ },
        { pattern: /border-(primary|secondary|accent)-(200|300|500)/ },
        // Flex
        { pattern: /flex-(row|col)/ },
        { pattern: /justify-(start|center|end|between)/ },
        { pattern: /items-(start|center|end)/ },
        // Widths
        { pattern: /w-(full|auto|1\/2|1\/3|1\/4|2\/3|3\/4)/ },
        // Opacity
        { pattern: /opacity-(0|50|75|100)/ },
        // Animations
        'animate-fade-in',
        'animate-slide-in-up',
        'animate-pulse-soft',
        'skeleton',
    ],
    darkMode: 'class',
    theme: {
        container: {
            center: true,
            padding: {
                DEFAULT: '1rem',
                sm: '1.5rem',
                lg: '2rem',
                xl: '2rem',
                '2xl': '2rem',
            },
        },
        extend: {
            screens: {
                xs: '420px',
            },
            maxWidth: {
                '8xl': '88rem',
                '9xl': '96rem',
                'full': '100%',
            },
            colors: {
                background: 'hsl(var(--background) / <alpha-value>)',
                foreground: 'hsl(var(--foreground) / <alpha-value>)',
                card: 'hsl(var(--card) / <alpha-value>)',
                border: 'hsl(var(--border) / <alpha-value>)',
                muted: 'hsl(var(--muted) / <alpha-value>)',
                // Brand primary - Warm terracotta/rust
                primary: {
                    DEFAULT: 'hsl(var(--primary) / <alpha-value>)',
                    50: '#fdf4f3',
                    100: '#fce7e4',
                    200: '#fbd3cd',
                    300: '#f7b3a9',
                    400: '#f08778',
                    500: '#e5604d',
                    600: '#d14430',
                    700: '#af3625',
                    800: '#913023',
                    900: '#792d23',
                    950: '#41140e',
                },
                // Neutral secondary - Stone grays
                secondary: {
                    DEFAULT: 'hsl(var(--secondary) / <alpha-value>)',
                    50: '#fafaf9',
                    100: '#f5f5f4',
                    200: '#e7e5e4',
                    300: '#d6d3d1',
                    400: '#a8a29e',
                    500: '#78716c',
                    600: '#57534e',
                    700: '#44403c',
                    800: '#292524',
                    900: '#1c1917',
                    950: '#0c0a09',
                },
                // Accent - Amber/gold
                accent: {
                    DEFAULT: 'hsl(var(--accent) / <alpha-value>)',
                    50: '#fffbeb',
                    100: '#fef3c7',
                    200: '#fde68a',
                    300: '#fcd34d',
                    400: '#fbbf24',
                    500: '#f59e0b',
                    600: '#d97706',
                    700: '#b45309',
                    800: '#92400e',
                    900: '#78350f',
                    950: '#451a03',
                },
                // Semantic colors
                success: {
                    DEFAULT: '#22c55e',
                    50: '#f0fdf4',
                    100: '#dcfce7',
                    200: '#bbf7d0',
                    300: '#86efac',
                    400: '#4ade80',
                    500: '#22c55e',
                    600: '#16a34a',
                    700: '#15803d',
                    800: '#166534',
                    900: '#14532d',
                    950: '#052e16',
                },
                warning: {
                    DEFAULT: '#f97316',
                    50: '#fff7ed',
                    100: '#ffedd5',
                    200: '#fed7aa',
                    300: '#fdba74',
                    400: '#fb923c',
                    500: '#f97316',
                    600: '#ea580c',
                    700: '#c2410c',
                    800: '#9a3412',
                    900: '#7c2d12',
                    950: '#431407',
                },
                error: {
                    DEFAULT: '#ef4444',
                    50: '#fef2f2',
                    100: '#fee2e2',
                    200: '#fecaca',
                    300: '#fca5a5',
                    400: '#f87171',
                    500: '#ef4444',
                    600: '#dc2626',
                    700: '#b91c1c',
                    800: '#991b1b',
                    900: '#7f1d1d',
                    950: '#450a0a',
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
                serif: ['Merriweather', 'Georgia', 'serif'],
                mono: ['JetBrains Mono', 'Menlo', 'monospace'],
                display: ['Poppins', 'system-ui', '-apple-system', 'sans-serif'],
            },
            fontSize: {
                '2xs': ['0.625rem', { lineHeight: '0.75rem' }],
            },
            spacing: {
                '18': '4.5rem',
                '88': '22rem',
                '128': '32rem',
            },
            borderRadius: {
                '4xl': '2rem',
            },
            zIndex: {
                '60': '60',
                '70': '70',
            },
            boxShadow: {
                'soft': '0 2px 15px -3px rgba(0, 0, 0, 0.07), 0 10px 20px -2px rgba(0, 0, 0, 0.04)',
                'soft-lg': '0 10px 40px -10px rgba(0, 0, 0, 0.1), 0 2px 20px -10px rgba(0, 0, 0, 0.06)',
                'inner-soft': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.04)',
            },
            animation: {
                'fade-in': 'fadeIn 0.3s ease-out',
                'fade-out': 'fadeOut 0.3s ease-out',
                'slide-in-up': 'slideInUp 0.3s ease-out',
                'slide-in-down': 'slideInDown 0.3s ease-out',
                'slide-in-left': 'slideInLeft 0.3s ease-out',
                'slide-in-right': 'slideInRight 0.3s ease-out',
                'scale-in': 'scaleIn 0.2s ease-out',
                'bounce-soft': 'bounceSoft 0.5s ease-out',
                'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
                'shimmer': 'shimmer 2s linear infinite',
            },
            keyframes: {
                fadeIn: {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' },
                },
                fadeOut: {
                    '0%': { opacity: '1' },
                    '100%': { opacity: '0' },
                },
                slideInUp: {
                    '0%': { transform: 'translateY(20px)', opacity: '0' },
                    '100%': { transform: 'translateY(0)', opacity: '1' },
                },
                slideInDown: {
                    '0%': { transform: 'translateY(-20px)', opacity: '0' },
                    '100%': { transform: 'translateY(0)', opacity: '1' },
                },
                slideInLeft: {
                    '0%': { transform: 'translateX(-20px)', opacity: '0' },
                    '100%': { transform: 'translateX(0)', opacity: '1' },
                },
                slideInRight: {
                    '0%': { transform: 'translateX(20px)', opacity: '0' },
                    '100%': { transform: 'translateX(0)', opacity: '1' },
                },
                scaleIn: {
                    '0%': { transform: 'scale(0.9)', opacity: '0' },
                    '100%': { transform: 'scale(1)', opacity: '1' },
                },
                bounceSoft: {
                    '0%, 100%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-5px)' },
                },
                pulseSoft: {
                    '0%, 100%': { opacity: '1' },
                    '50%': { opacity: '0.7' },
                },
                shimmer: {
                    '0%': { backgroundPosition: '-200% 0' },
                    '100%': { backgroundPosition: '200% 0' },
                },
            },
            transitionTimingFunction: {
                'bounce-in': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
            },
            aspectRatio: {
                'product': '4 / 5',
                'hero': '21 / 9',
            },
            typography: (theme) => ({
                DEFAULT: {
                    css: {
                        color: theme('colors.secondary.800'),
                        a: {
                            color: theme('colors.primary.600'),
                            '&:hover': {
                                color: theme('colors.primary.700'),
                            },
                        },
                        h1: { color: theme('colors.secondary.900') },
                        h2: { color: theme('colors.secondary.900') },
                        h3: { color: theme('colors.secondary.900') },
                        h4: { color: theme('colors.secondary.900') },
                    },
                },
            }),
        },
    },
    plugins: [
        require('@tailwindcss/forms')({
            strategy: 'class',
        }),
        require('@tailwindcss/typography'),
        require('@tailwindcss/aspect-ratio'),
        // Custom plugin for utilities
        function({ addUtilities, addComponents, theme }) {
            // Text gradient
            addUtilities({
                '.text-gradient': {
                    'background-clip': 'text',
                    '-webkit-background-clip': 'text',
                    '-webkit-text-fill-color': 'transparent',
                },
                '.text-gradient-primary': {
                    'background-image': `linear-gradient(135deg, ${theme('colors.primary.500')}, ${theme('colors.accent.500')})`,
                },
            });
            
            // Scrollbar styles
            addUtilities({
                '.scrollbar-hide': {
                    '-ms-overflow-style': 'none',
                    'scrollbar-width': 'none',
                    '&::-webkit-scrollbar': {
                        display: 'none',
                    },
                },
                '.scrollbar-thin': {
                    'scrollbar-width': 'thin',
                    '&::-webkit-scrollbar': {
                        width: '6px',
                        height: '6px',
                    },
                    '&::-webkit-scrollbar-track': {
                        background: theme('colors.secondary.100'),
                        borderRadius: '3px',
                    },
                    '&::-webkit-scrollbar-thumb': {
                        background: theme('colors.secondary.300'),
                        borderRadius: '3px',
                        '&:hover': {
                            background: theme('colors.secondary.400'),
                        },
                    },
                },
            });
            
            // Skeleton loading
            addComponents({
                '.skeleton': {
                    background: `linear-gradient(90deg, ${theme('colors.secondary.100')} 25%, ${theme('colors.secondary.200')} 50%, ${theme('colors.secondary.100')} 75%)`,
                    backgroundSize: '200% 100%',
                    animation: 'shimmer 2s linear infinite',
                    borderRadius: theme('borderRadius.md'),
                },
            });
        },
    ],
};
