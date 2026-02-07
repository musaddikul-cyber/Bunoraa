/**
 * Shipping API Module
 * @module api/shipping
 */

const ShippingApi = (function() {
    'use strict';

    async function getZones() {
        return ApiClient.get('/shipping/zones/', {}, { useCache: true, cacheTTL: 300000 });
    }

    async function getRates(params = {}) {
        return ApiClient.get('/shipping/rates/', {
            zone_id: params.zoneId,
            weight: params.weight,
            subtotal: params.subtotal
        });
    }

    async function getMethods() {
        return ApiClient.get('/shipping/methods/', {}, { useCache: true, cacheTTL: 300000 });
    }

    async function calculateShipping(data) {
        return ApiClient.post('/shipping/calculate/', data);
    }

    async function getDeliveryEstimate(data) {
        return ApiClient.post('/shipping/estimate/', data);
    }

    async function trackShipment(trackingNumber) {
        return ApiClient.get(`/shipping/track/${trackingNumber}/`);
    }

    return {
        getZones,
        getRates,
        getMethods,
        calculateShipping,
        getDeliveryEstimate,
        trackShipment
    };
})();

window.ShippingApi = ShippingApi;
