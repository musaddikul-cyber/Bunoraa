# Product Card Patterns: Research Notes

## Objective
Create a reusable set of product card patterns for different ecommerce surfaces while aligning to UX, accessibility, and performance guidance.

## Primary findings
- Product listing UX quality has a direct effect on findability; many sites still underperform and need significant listing design improvements.
- Ratings are high-value in list scanning, but users need both average score and rating count to trust and compare products.
- Card components should be modular, actionable, and self-contained (not decorative wrappers).
- Image-heavy card layouts should use lazy loading and async decoding for better performance.
- Interactive controls should respect minimum target size guidance and spacing.
- Modern commerce themes expose product-card options such as secondary image, badges, vendor/rating display, and quick-add behavior.

## Sources
- Baymard product lists benchmark: https://baymard.com/research/ecommerce-product-lists
- Baymard product finding update: https://baymard.com/blog/product-finding-2024-launch
- Baymard ratings count in list items: https://baymard.com/blog/user-perception-of-product-ratings
- Baymard rating sort logic: https://baymard.com/blog/sort-by-customer-ratings
- USWDS card guidance: https://designsystem.digital.gov/components/card
- web.dev image performance and lazy loading: https://web.dev/learn/performance/image-performance
- web.dev browser-level lazy loading guidance: https://web.dev/articles/browser-level-lazy-loading-for-cmss
- WCAG 2.2 Target Size (Minimum): https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum
- Ant Design card component patterns: https://ant.design/components/card
- Shopify Dawn product card snippet: https://raw.githubusercontent.com/Shopify/dawn/main/snippets/card-product.liquid

## Variant mapping
- `standard`: baseline listing card for category/search grids.
- `compact`: reduced footprint for side rails and mobile strips.
- `horizontal`: scan-friendly list row for list-mode results.
- `overlay`: visual-first storytelling block for editorial placements.
- `deal`: discount-forward promotion card for sale modules.
- `quick-add`: conversion-first card for fast cart actions.
- `minimal`: quiet pattern for dense mixed-content layouts.
- `editorial`: narrative presentation for artisan/collection spotlights.
- `rating-focus`: trust-first card for review-sensitive products.
- `compare-focus`: decision-support card for shortlist workflows.
- `inventory-focus`: stock/urgency card for limited availability placements.
- `dense-row`: high-density row for power-user result surfaces.

