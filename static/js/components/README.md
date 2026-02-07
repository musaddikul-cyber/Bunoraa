# Pure JavaScript UI Component Library

A comprehensive collection of **60+ production-ready UI components** built with pure JavaScript (ES6+) and Tailwind CSS. No external frameworks or dependencies required.

## ðŸ“¦ What's Included

### Tier 1: Basic Components (8)
- **Button** - Versatile button with multiple variants (primary, secondary, destructive, outline, ghost) and sizes
- **Badge** - Status badges with color variants
- **Label** - Form label with optional required indicator
- **Separator** - Horizontal/vertical divider component
- **Typography** - Semantic text component (h1-h6, body, code, muted)
- **Kbd** - Keyboard key display component
- **Spinner** - Loading spinner with size and color options
- **Item** - Generic list item with selection state

### Tier 2: Input Components (8)
- **Input** - Text input with validation and change handlers
- **InputGroup** - Wrapper for inputs with prefix/suffix elements
- **Textarea** - Multi-line text input with resize control
- **Checkbox** - Checkbox with label integration
- **RadioGroup** - Radio button group with orientation options
- **Toggle** - Toggle button with pressed state
- **Switch** - iOS-style toggle switch
- **NativeSelect** - Dropdown select with dynamic options

### Tier 3: Container Components (6)
- **Card** - Container with header, content, footer sections
- **Avatar** - Image or initial-based avatar with fallback
- **Alert** - Alert box with type variants (success, warning, error, info)
- **Skeleton** - Loading skeleton with count support
- **Progress** - Progress bar with value and animation
- **Slider** - Range input with custom styling

### Tier 4: Interactive Components (5)
- **ButtonGroup** - Grouped buttons with border separation
- **Tabs** - Tab interface with pills/default variants
- **Breadcrumb** - Navigation breadcrumb trail
- **ToggleGroup** - Toggle button group (single/multi-select)
- **Pagination** - Pagination controls with page navigation

### Tier 5: Complex Modal Components (9)
    - **Dialog** - Modal dialog with backdrop and focus trap
    - **AlertDialog** - Confirmation dialog
    - **Tooltip** - Hover/focus tooltip with position options
    - **Popover** - Click-triggered popover
    - **Drawer** - Side drawer with slide animation
    - **Sheet** - Bottom sheet modal
    - **HoverCard** - Card appearing on hover
    - **ContextMenu** - Right-click context menu
    - **DropdownMenu** - Click dropdown menu

### Tier 6: Advanced Components (6)
    - **Select** - Searchable custom select with filtering
    - **Calendar** - Interactive calendar with month navigation
    - **InputOTP** - OTP input with auto-advance
    - **DatePicker** - Date picker with format support
    - **Combobox** - Searchable dropdown with filtering
    - **Command** - Command palette/search interface

### Tier 7: Specialized Components (5)
    - **Collapsible** - Expandable/collapsible section with animation
    - **Carousel** - Image carousel with autoplay and navigation
    - **AspectRatio** - Container maintaining aspect ratio
    - **ScrollArea** - Custom scrollable container
    - **Sidebar** - Navigation sidebar with collapse option

### Tier 8: Data & Form Components (5)
    - **Empty** - Empty state display with icon/message/action
    - **Form** - Form wrapper with field validation
    - **DataTable** - Table with sorting, selection, and row management
    - **Toast** - Toast notifications with auto-dismiss
    - **Chart** - Simple SVG bar/line chart visualization

## ðŸš€ Getting Started

### Installation

1. Copy the `components` folder to your project's static/js directory:
```
static/js/components/
â”œâ”€â”€ BaseComponent.js
â”œâ”€â”€ utils.js
â”œâ”€â”€ Button.js
â”œâ”€â”€ Badge.js
â”œâ”€â”€ ... (60+ component files)
â”œâ”€â”€ index.js
â””â”€â”€ README.md
```

2. Import components in your JavaScript:

```javascript
// Import all components
import * as Components from './components/index.js';

// Or import specific components
import { Button, Card, Input } from './components/index.js';
```

### Basic Usage

```javascript
// Create a button
const button = new Button({
    label: 'Click Me',
    variant: 'primary',
    size: 'md',
    onClick: () => alert('Clicked!')
}).create();

document.body.appendChild(button);
```

## ðŸ“‹ Architecture

### BaseComponent

All components inherit from `BaseComponent`, providing common functionality:

```javascript
class MyComponent extends BaseComponent {
    constructor(options) {
        super(options);
        this.options = options;
    }
    
    create() {
        const element = this.createElement('div');
        // Your component logic
        return element;
    }
}
```

#### BaseComponent Methods

- `create(tag, options)` - Factory method to create DOM elements
- `mount(selector)` - Attach component to DOM
- `on(event, handler)` - Register event listener with automatic cleanup
- `attr(name, value)` - Get/set element attributes
- `addClass(className)` / `removeClass(className)` - Class manipulation
- `show()` / `hide()` / `toggle()` - Visibility control
- `destroy()` - Cleanup and removal

### Utility Functions

The `utils.js` file provides 20+ helper functions:

```javascript
import { clsx, createElement, debounce, throttle, createFocusTrap } from './utils.js';

// Class name merging
const className = clsx('px-4', condition && 'py-2', 'rounded');

// DOM creation
const el = createElement('button', { className: 'btn', textContent: 'Click' });

// Performance optimization
const handleResize = debounce(() => console.log('Resized'), 300);
window.addEventListener('resize', handleResize);

// Focus management
const focusTrap = createFocusTrap(modalElement);
focusTrap.activate();
```

## ðŸŽ¨ Styling with Tailwind

All components use Tailwind CSS utility classes. No CSS file neededâ€”classes are applied programmatically:

```javascript
const styles = {
    base: 'px-4 py-2 rounded font-medium transition-colors duration-200',
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300'
};
```

## ðŸ“– Component API Examples

### Button Component

```javascript
const button = new Button({
    label: 'Submit',
    variant: 'primary', // 'primary' | 'secondary' | 'destructive' | 'outline' | 'ghost'
    size: 'md', // 'sm' | 'md' | 'lg'
    disabled: false,
    onClick: (event) => {}
}).create();
```

### Card Component

```javascript
const card = new Card({
    title: 'Card Title',
    subtitle: 'Card Subtitle',
    content: 'Card content goes here',
    footer: 'Footer content',
    hoverable: true
}).create();
```

### Dialog Component

```javascript
const dialog = new Dialog({
    title: 'Confirm Action',
    content: 'Are you sure?',
    closeButton: true,
    closeOnBackdrop: true,
    onClose: () => console.log('Dialog closed')
});

dialog.open();
dialog.close();
```

### Form Component

```javascript
const form = new Form({
    fields: [
        { name: 'email', label: 'Email', type: 'email', required: true },
        { name: 'message', label: 'Message', type: 'textarea' }
    ],
    onSubmit: (values) => console.log(values)
}).create();
```

### DataTable Component

```javascript
const table = new DataTable({
    columns: ['Name', 'Email', 'Status'],
    rows: [
        { Name: 'John', Email: 'john@example.com', Status: 'Active' },
        { Name: 'Jane', Email: 'jane@example.com', Status: 'Inactive' }
    ],
    sortable: true,
    selectable: true
}).create();

const selectedRows = table.getSelectedRows();
```

## ðŸ”§ Common Patterns

### Event Handling

```javascript
const button = new Button({ label: 'Click' });
const el = button.create();

// Register event listener
button.on('click', (event) => {
    console.log('Button clicked');
});

// Multiple listeners
button.on('mouseover', () => el.classList.add('hovered'));
button.on('mouseout', () => el.classList.remove('hovered'));
```

### Conditional Rendering

```javascript
const alert = new Alert({
    type: 'success', // 'success' | 'warning' | 'error' | 'info'
    content: 'Operation completed!',
    closeable: true
});
```

### Dynamic Updates

```javascript
const progress = new Progress({ value: 30 });
const el = progress.create();

// Update value
progress.setValue(75);

// Increment/Decrement
progress.increment(10);
progress.decrement(5);
```

### Nested Components

```javascript
const card = new Card({
    title: 'User Profile',
    content: 'Profile information here'
});

const button = new Button({ label: 'Save' });

// Append button to card
card.content.appendChild(button.create());
```

## ðŸŽ¯ Best Practices

1. **Always call `.create()`** to get the DOM element
2. **Cleanup with `.destroy()`** to prevent memory leaks
3. **Use event delegation** for better performance with many components
4. **Leverage BaseComponent methods** for consistent behavior
5. **Reuse utility functions** from utils.js
6. **Apply consistent styling** through className options
7. **Handle async operations** with proper state management

## ðŸŽ¨ Customization

### Custom Styling

All components accept a `className` parameter for additional styles:

```javascript
const button = new Button({
    label: 'Custom',
    className: 'shadow-lg border-2 border-blue-400'
}).create();
```

### Extending Components

```javascript
class CustomButton extends Button {
    create() {
        const element = super.create();
        // Add custom logic
        element.style.textTransform = 'uppercase';
        return element;
    }
}
```

### Theme Customization

Modify the CSS utility object in `utils.js`:

```javascript
const css = {
    colors: {
        primary: 'bg-indigo-600',
        secondary: 'bg-purple-600'
    }
};
```

## ðŸ“± Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## â™¿ Accessibility

Components include built-in accessibility features:

- ARIA labels and roles
- Keyboard navigation support
- Focus management
- Semantic HTML
- Color contrast compliance

## ðŸ§ª Testing

Example test structure:

```javascript
// test.js
import { Button } from './components/index.js';

const button = new Button({ label: 'Test' });
const element = button.create();

console.assert(element.textContent === 'Test', 'Button label should be Test');
console.assert(element.className.includes('rounded'), 'Should have rounded class');
```

## ðŸ”„ Component Lifecycle

```javascript
// 1. Create instance
const component = new Button({ label: 'Click' });

// 2. Create DOM element
const element = component.create();

// 3. Mount to document
document.body.appendChild(element);

// 4. Interact with component
component.on('click', handler);
component.setAttribute('disabled', true);

// 5. Clean up
component.destroy();
```

## ðŸ“š File Structure

```
static/js/components/
â”œâ”€â”€ BaseComponent.js          # Base class for all components
â”œâ”€â”€ utils.js                  # 20+ utility functions
â”œâ”€â”€ Button.js                 # Button component
â”œâ”€â”€ Badge.js                  # Badge component
â”œâ”€â”€ ... (50+ more components)
â”œâ”€â”€ index.js                  # Central exports
â”œâ”€â”€ DEMO.html                 # Interactive demo
â””â”€â”€ README.md                 # This file
```

## ðŸš€ Performance Tips

1. **Lazy load components** - Only import what you need
2. **Use event delegation** - Attach listeners to parent elements
3. **Debounce/throttle** - Limit rapid event handlers
4. **Destroy unused components** - Call `.destroy()` when done
5. **Batch DOM updates** - Minimize reflows and repaints

## ðŸ› Troubleshooting

### Component not rendering
```javascript
// Make sure to call create() and append
const button = new Button({ label: 'Test' });
const element = button.create();
document.body.appendChild(element);
```

### Event listeners not firing
```javascript
// Use the component's on() method, not addEventListener
button.on('click', handler);
// Don't use: button.element.addEventListener('click', handler);
```

### Styling issues
```javascript
// Ensure Tailwind CSS is loaded
// Check className values in browser dev tools
console.log(element.className);
```

## ðŸ“„ License

MIT License - Feel free to use in your projects

## ðŸ¤ Contributing

To extend the library:

1. Create a new component extending BaseComponent
2. Follow the naming conventions
3. Add to index.js
4. Test thoroughly
5. Document usage

## ðŸ“ž Support

For issues or questions, refer to the inline code comments and DEMO.html for examples.

---

**Built with â¤ï¸ using pure JavaScript and Tailwind CSS**
