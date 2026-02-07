/**
 * CategoryCard Component
 */

export function CategoryCard(category) {
    // Create anchor element
    const card = document.createElement('a');
    card.href = `/categories/${category.slug}/`;
    card.className = 'group block';

    // Card inner (image or fallback)
    const cardInner = document.createElement('div');
    cardInner.className = 'relative aspect-square rounded-xl overflow-hidden bg-gray-100';

    // Image or fallback
    let imgSrc = '';
    // API provides `image_url` by default; prefer that
    if (typeof category.image_url === 'string' && category.image_url) imgSrc = category.image_url;
    else if (typeof category.image === 'string' && category.image) imgSrc = category.image;
    else if (category.image && typeof category.image === 'object') imgSrc = category.image.url || category.image.src || category.image_url || '';
    else if (typeof category.banner_image === 'string' && category.banner_image) imgSrc = category.banner_image;
    else if (category.banner_image && typeof category.banner_image === 'object') imgSrc = category.banner_image.url || category.banner_image.src || '';
    else if (typeof category.hero_image === 'string' && category.hero_image) imgSrc = category.hero_image;
    else if (category.hero_image && typeof category.hero_image === 'object') imgSrc = category.hero_image.url || category.hero_image.src || '';
    else if (typeof category.thumbnail === 'string' && category.thumbnail) imgSrc = category.thumbnail;
    else if (category.thumbnail && typeof category.thumbnail === 'object') imgSrc = category.thumbnail.url || category.thumbnail.src || '';



    if (imgSrc) {
        const img = document.createElement('img');
        img.src = imgSrc;
        img.alt = category.name || '';
        img.className = 'w-full h-full object-cover group-hover:scale-105 transition-transform duration-300';
        img.loading = 'lazy';
        img.decoding = 'async';
        // If image fails to load, show fallback block instead
        img.onerror = (err) => {
            try {
                img.remove();
                const fallback = document.createElement('div');
                fallback.className = 'w-full h-full flex items-center justify-center bg-gradient-to-br from-primary-100 to-primary-200';
                fallback.innerHTML = `<svg class="w-12 h-12 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/></svg>`;
                cardInner.appendChild(fallback);
            } catch (e) {}
        }; 
        cardInner.appendChild(img);
    } else {
        const fallback = document.createElement('div');
        fallback.className = 'w-full h-full flex items-center justify-center bg-gradient-to-br from-primary-100 to-primary-200';
        fallback.innerHTML = `<svg class="w-12 h-12 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/></svg>`;
        cardInner.appendChild(fallback);
    }

    const overlay = document.createElement('div');
    // Lighter overlay on light mode, stronger on dark
    overlay.className = 'absolute inset-0 bg-gradient-to-t from-black/30 dark:from-black/60 to-transparent';
    cardInner.appendChild(overlay);
    card.appendChild(cardInner);

    const name = document.createElement('h3');
    name.className = 'mt-3 text-sm font-medium text-stone-900 group-hover:text-primary-600 transition-colors text-center dark:text-white';
    name.textContent = category.name;
    card.appendChild(name);

    if (category.product_count) {
        const count = document.createElement('p');
        count.className = 'text-xs text-stone-600 dark:text-white/60 text-center';
        count.textContent = `${category.product_count} products`;
        card.appendChild(count);
    }

    return card;
}
