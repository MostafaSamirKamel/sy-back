import { prisma } from '../lib/prisma.js';
import { mergeStationConfig, parsePartialStationConfig, parseStationConfig, serializeStationConfig, } from '../lib/stationConfig.js';
import { resolveUserUniversityId } from '../lib/universityScope.js';
export async function resolveStationConfigForUser(caseId, userId) {
    const caseData = await prisma.case.findUnique({
        where: { id: caseId },
        select: { stationConfig: true },
    });
    if (!caseData) {
        return parseStationConfig(null);
    }
    const base = parseStationConfig(caseData.stationConfig);
    const universityId = await resolveUserUniversityId(userId);
    if (!universityId)
        return base;
    const override = await prisma.caseUniversityOverride.findFirst({
        where: { caseId, universityId, isActive: true },
        select: { stationConfig: true },
    });
    if (!override)
        return base;
    return mergeStationConfig(base, parsePartialStationConfig(override.stationConfig));
}
export async function serializeResolvedStationConfigForUser(caseId, userId) {
    const config = await resolveStationConfigForUser(caseId, userId);
    return serializeStationConfig(config);
}
