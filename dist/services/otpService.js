import bcrypt from 'bcryptjs';
import { prisma } from '../lib/prisma.js';
import { sendOtpEmail, isSmtpConfigured } from './emailService.js';
import { normalizeEmailLang } from './emailTemplates.js';
const OTP_TTL_MS = 15 * 60 * 1000;
function generateOtpCode() {
    return String(Math.floor(100000 + Math.random() * 900000));
}
async function resolveEmailLang(userId, lang) {
    if (lang)
        return normalizeEmailLang(lang);
    const user = await prisma.user.findUnique({
        where: { id: userId },
        select: { preferredLang: true },
    });
    return normalizeEmailLang(user?.preferredLang || 'en');
}
export async function issueAndSendOtp(userId, email, firstName, lang) {
    if (!isSmtpConfigured()) {
        throw new Error('SMTP_NOT_CONFIGURED');
    }
    const emailLang = await resolveEmailLang(userId, lang);
    const code = generateOtpCode();
    const otpCode = await bcrypt.hash(code, 10);
    const otpExpires = new Date(Date.now() + OTP_TTL_MS);
    await prisma.user.update({
        where: { id: userId },
        data: {
            otpCode,
            otpExpires,
            ...(lang ? { preferredLang: emailLang } : {}),
        },
    });
    await sendOtpEmail(email, code, firstName, emailLang);
}
export async function verifyUserOtp(email, code) {
    const user = await prisma.user.findUnique({ where: { email } });
    if (!user || !user.otpCode || !user.otpExpires) {
        return { ok: false, reason: 'INVALID' };
    }
    if (user.otpExpires < new Date()) {
        return { ok: false, reason: 'EXPIRED' };
    }
    const valid = await bcrypt.compare(code, user.otpCode);
    if (!valid) {
        return { ok: false, reason: 'INVALID' };
    }
    const updated = await prisma.user.update({
        where: { id: user.id },
        data: { emailVerified: true, otpCode: null, otpExpires: null },
    });
    return { ok: true, user: updated };
}
