import crypto from 'crypto';
const PAYMOB_BASE = 'https://accept.paymob.com/api';
export function isPaymobConfigured() {
    return !!(process.env.PAYMOB_API_KEY &&
        process.env.PAYMOB_INTEGRATION_ID &&
        process.env.PAYMOB_IFRAME_ID &&
        process.env.PAYMOB_HMAC_SECRET);
}
async function paymobAuthToken() {
    const apiKey = process.env.PAYMOB_API_KEY;
    const res = await fetch(`${PAYMOB_BASE}/auth/tokens`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: apiKey }),
    });
    if (!res.ok) {
        throw new Error(`Paymob auth failed (${res.status})`);
    }
    const data = (await res.json());
    if (!data.token)
        throw new Error('Paymob auth returned no token');
    return data.token;
}
export async function createPaymobCheckout(params) {
    const authToken = await paymobAuthToken();
    const amountCents = params.amountEgp * 100;
    const currency = process.env.PAYMOB_CURRENCY || 'EGP';
    const integrationId = Number(process.env.PAYMOB_INTEGRATION_ID);
    const iframeId = process.env.PAYMOB_IFRAME_ID;
    const orderRes = await fetch(`${PAYMOB_BASE}/ecommerce/orders`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            auth_token: authToken,
            delivery_needed: false,
            amount_cents: amountCents,
            currency,
            merchant_order_id: params.merchantOrderId,
            items: [
                {
                    name: 'Synoza subscription',
                    amount_cents: amountCents,
                    description: params.merchantOrderId,
                    quantity: 1,
                },
            ],
        }),
    });
    if (!orderRes.ok) {
        throw new Error(`Paymob order failed (${orderRes.status})`);
    }
    const order = (await orderRes.json());
    const phone = params.billing.phone?.replace(/\D/g, '') || '01000000000';
    const keyRes = await fetch(`${PAYMOB_BASE}/acceptance/payment_keys`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            auth_token: authToken,
            amount_cents: amountCents,
            expiration: 3600,
            order_id: order.id,
            billing_data: {
                apartment: 'NA',
                email: params.billing.email,
                floor: 'NA',
                first_name: params.billing.firstName || 'Student',
                street: 'NA',
                building: 'NA',
                phone_number: phone,
                shipping_method: 'NA',
                postal_code: 'NA',
                city: 'Cairo',
                country: 'EG',
                last_name: params.billing.lastName || 'Synoza',
                state: 'NA',
            },
            currency,
            integration_id: integrationId,
        }),
    });
    if (!keyRes.ok) {
        throw new Error(`Paymob payment key failed (${keyRes.status})`);
    }
    const keyData = (await keyRes.json());
    if (!keyData.token)
        throw new Error('Paymob payment key missing');
    return {
        checkoutUrl: `${PAYMOB_BASE}/acceptance/iframes/${iframeId}?payment_token=${keyData.token}`,
        providerOrderId: String(order.id),
    };
}
function normalizeHmacValue(value) {
    if (value === true)
        return 'true';
    if (value === false)
        return 'false';
    return String(value ?? '');
}
export function verifyPaymobTransactionHmac(transaction, receivedHmac) {
    const secret = process.env.PAYMOB_HMAC_SECRET;
    if (!secret || !receivedHmac)
        return false;
    const parts = [
        transaction.amount_cents,
        transaction.created_at,
        transaction.currency,
        transaction.error_occured,
        transaction.has_parent_transaction,
        transaction.id,
        transaction.integration_id,
        transaction.is_3d_secure,
        transaction.is_auth,
        transaction.is_capture,
        transaction.is_refunded,
        transaction.is_standalone_payment,
        transaction.is_voided,
        transaction.order?.id,
        transaction.owner,
        transaction.pending,
        transaction.source_data?.pan,
        transaction.source_data?.sub_type,
        transaction.source_data?.type,
        transaction.success,
    ].map(normalizeHmacValue);
    const calculated = crypto.createHmac('sha512', secret).update(parts.join('')).digest('hex');
    return calculated === receivedHmac;
}
export function isPaymobSuccess(value) {
    return value === true || value === 'true';
}
