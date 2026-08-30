import jwt from 'jsonwebtoken';
const REFRESH_GRACE_SECONDS = 90 * 24 * 60 * 60;
export function verifyAccessToken(token) {
    const secret = process.env.JWT_SECRET;
    if (!secret)
        return null;
    try {
        return jwt.verify(token, secret);
    }
    catch {
        return null;
    }
}
/** Accepts valid tokens and recently expired tokens (for silent refresh). */
export function verifyTokenForRefresh(token) {
    const secret = process.env.JWT_SECRET;
    if (!secret)
        return null;
    try {
        return jwt.verify(token, secret);
    }
    catch (err) {
        if (!(err instanceof jwt.TokenExpiredError))
            return null;
        const decoded = jwt.decode(token);
        if (!decoded?.exp)
            return null;
        if (Date.now() / 1000 - decoded.exp > REFRESH_GRACE_SECONDS)
            return null;
        try {
            return jwt.verify(token, secret, { ignoreExpiration: true });
        }
        catch {
            return null;
        }
    }
}
export function authenticate(req, res, next) {
    const header = req.headers.authorization;
    if (!header?.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'Authentication required' });
    }
    const payload = verifyAccessToken(header.slice(7));
    if (!payload) {
        return res.status(401).json({ error: 'Invalid or expired token' });
    }
    req.user = payload;
    void import('../services/lastSeenService.js').then(({ touchLastSeen }) => {
        touchLastSeen(payload.id);
    });
    next();
}
export function authorize(...roles) {
    return (req, res, next) => {
        if (!req.user) {
            return res.status(401).json({ error: 'Authentication required' });
        }
        if (roles.length && !roles.includes(req.user.role)) {
            return res.status(403).json({ error: 'Insufficient permissions' });
        }
        next();
    };
}
