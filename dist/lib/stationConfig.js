export const ALL_MANEUVERS = ['inspection', 'palpation', 'percussion', 'auscultation'];
export const MAIN_STAGES = ['history', 'examination', 'investigations', 'diagnosis'];
export const DEFAULT_PATIENT_BEHAVIOR = {
    instructions: '',
    tone: '',
    emotion: '',
    preferredLanguage: 'AUTO',
    constraints: '',
};
export const DEFAULT_MANEUVER_OPENING_TEMPLATE = 'I am evaluating your clinical {{maneuver}}. Take a close look at the clinical presentation and images provided. Describe your findings systematically and explain what you would look for during {{maneuver}}, including any scars, deformities, or visible abnormalities.';
export const MANEUVER_LABELS = {
    inspection: { en: 'Inspection', ar: 'الفحص البصري' },
    palpation: { en: 'Palpation', ar: 'الجس' },
    percussion: { en: 'Percussion', ar: 'النقر' },
    auscultation: { en: 'Auscultation', ar: 'الاستماع' },
};
export const DEFAULT_STATION_CONFIG = {
    enabledManeuvers: [...ALL_MANEUVERS],
    enableHistoryExaminer: true,
    enableInvestigations: true,
    stageOrder: [...MAIN_STAGES],
    maneuverOpeningMessages: {},
    maneuverLabels: {},
    patientBehavior: { ...DEFAULT_PATIENT_BEHAVIOR },
};
function parseManeuverOpeningMessages(raw) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw))
        return {};
    const result = {};
    for (const maneuver of ALL_MANEUVERS) {
        const value = raw[maneuver];
        if (typeof value === 'string' && value.trim()) {
            result[maneuver] = value.trim();
        }
    }
    return result;
}
function parseManeuverLabels(raw) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw))
        return {};
    const result = {};
    for (const maneuver of ALL_MANEUVERS) {
        const value = raw[maneuver];
        if (!value || typeof value !== 'object' || Array.isArray(value))
            continue;
        const en = String(value.en ?? '').trim();
        const ar = String(value.ar ?? '').trim();
        if (en || ar)
            result[maneuver] = { en: en || MANEUVER_LABELS[maneuver].en, ar: ar || MANEUVER_LABELS[maneuver].ar };
    }
    return result;
}
function isPreferredLanguage(value) {
    return value === 'AUTO' || value === 'AR' || value === 'EN';
}
export function parsePatientBehavior(raw) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
        return { ...DEFAULT_PATIENT_BEHAVIOR };
    }
    const obj = raw;
    return {
        instructions: typeof obj.instructions === 'string' ? obj.instructions.trim() : '',
        tone: typeof obj.tone === 'string' ? obj.tone.trim() : '',
        emotion: typeof obj.emotion === 'string' ? obj.emotion.trim() : '',
        preferredLanguage: isPreferredLanguage(obj.preferredLanguage)
            ? obj.preferredLanguage
            : 'AUTO',
        constraints: typeof obj.constraints === 'string' ? obj.constraints.trim() : '',
    };
}
export function mergePatientBehavior(base, override) {
    if (!override)
        return { ...base };
    return {
        instructions: override.instructions !== undefined ? String(override.instructions).trim() : base.instructions,
        tone: override.tone !== undefined ? String(override.tone).trim() : base.tone,
        emotion: override.emotion !== undefined ? String(override.emotion).trim() : base.emotion,
        preferredLanguage: isPreferredLanguage(override.preferredLanguage)
            ? override.preferredLanguage
            : base.preferredLanguage,
        constraints: override.constraints !== undefined ? String(override.constraints).trim() : base.constraints,
    };
}
export function formatPatientBehaviorPrompt(behavior) {
    if (!behavior)
        return '';
    const lines = [];
    if (behavior.tone)
        lines.push(`- Tone: ${behavior.tone}`);
    if (behavior.emotion)
        lines.push(`- Emotion: ${behavior.emotion}`);
    if (behavior.preferredLanguage && behavior.preferredLanguage !== 'AUTO') {
        lines.push(`- Preferred language: ${behavior.preferredLanguage}`);
    }
    if (behavior.instructions)
        lines.push(`- Custom instructions:\n${behavior.instructions}`);
    if (behavior.constraints)
        lines.push(`- Hard constraints (must follow):\n${behavior.constraints}`);
    if (!lines.length)
        return '';
    return `\nSTATION PATIENT BEHAVIOR (admin overrides — follow these):\n${lines.join('\n')}\n`;
}
function patientBehaviorHasContent(behavior) {
    return !!(behavior.instructions ||
        behavior.tone ||
        behavior.emotion ||
        behavior.constraints ||
        (behavior.preferredLanguage && behavior.preferredLanguage !== 'AUTO'));
}
function mergeManeuverOpeningMessages(base, override) {
    if (!override)
        return { ...base };
    return { ...base, ...override };
}
function mergeManeuverLabels(base, override) {
    if (!override)
        return { ...base };
    return { ...base, ...override };
}
export function resolveManeuverLabel(maneuverId, config, lang = 'en') {
    const id = isManeuverId(maneuverId) ? maneuverId : 'inspection';
    const custom = config?.maneuverLabels?.[id];
    if (custom) {
        const preferred = lang === 'ar' ? custom.ar : custom.en;
        if (preferred?.trim())
            return preferred.trim();
        if (custom.en?.trim())
            return custom.en.trim();
        if (custom.ar?.trim())
            return custom.ar.trim();
    }
    return lang === 'ar' ? MANEUVER_LABELS[id].ar : MANEUVER_LABELS[id].en;
}
export function defaultManeuverOpeningMessage(maneuverId, config) {
    const name = resolveManeuverLabel(maneuverId, config, 'en');
    return DEFAULT_MANEUVER_OPENING_TEMPLATE.replace(/\{\{maneuver\}\}/g, name);
}
export function resolveManeuverOpeningMessage(maneuverId, config) {
    const id = maneuverId;
    const custom = config.maneuverOpeningMessages[id]?.trim();
    if (custom)
        return custom;
    if (isManeuverId(maneuverId))
        return defaultManeuverOpeningMessage(maneuverId, config);
    return defaultManeuverOpeningMessage('inspection', config);
}
function isManeuverId(value) {
    return typeof value === 'string' && ALL_MANEUVERS.includes(value);
}
function isMainStage(value) {
    return typeof value === 'string' && MAIN_STAGES.includes(value);
}
function normalizeStageOrder(order) {
    const seen = new Set();
    const normalized = [];
    for (const stage of order ?? MAIN_STAGES) {
        if (!isMainStage(stage) || seen.has(stage))
            continue;
        seen.add(stage);
        normalized.push(stage);
    }
    for (const stage of MAIN_STAGES) {
        if (!seen.has(stage))
            normalized.push(stage);
    }
    return normalized;
}
export function parseStationConfig(raw) {
    if (!raw?.trim()) {
        return {
            ...DEFAULT_STATION_CONFIG,
            enabledManeuvers: [...ALL_MANEUVERS],
            stageOrder: [...MAIN_STAGES],
            patientBehavior: { ...DEFAULT_PATIENT_BEHAVIOR },
        };
    }
    try {
        const parsed = JSON.parse(raw);
        const enabled = Array.isArray(parsed.enabledManeuvers)
            ? parsed.enabledManeuvers.filter(isManeuverId)
            : [...ALL_MANEUVERS];
        return {
            enabledManeuvers: enabled.length > 0 ? enabled : [...ALL_MANEUVERS],
            enableHistoryExaminer: parsed.enableHistoryExaminer !== false,
            enableInvestigations: parsed.enableInvestigations !== false,
            stageOrder: normalizeStageOrder(Array.isArray(parsed.stageOrder) ? parsed.stageOrder.filter(isMainStage) : undefined),
            maneuverOpeningMessages: parseManeuverOpeningMessages(parsed.maneuverOpeningMessages),
            maneuverLabels: parseManeuverLabels(parsed.maneuverLabels),
            patientBehavior: parsePatientBehavior(parsed.patientBehavior),
        };
    }
    catch {
        return {
            ...DEFAULT_STATION_CONFIG,
            enabledManeuvers: [...ALL_MANEUVERS],
            stageOrder: [...MAIN_STAGES],
            maneuverOpeningMessages: {},
            maneuverLabels: {},
            patientBehavior: { ...DEFAULT_PATIENT_BEHAVIOR },
        };
    }
}
export function parsePartialStationConfig(raw) {
    if (!raw?.trim())
        return {};
    try {
        const parsed = JSON.parse(raw);
        const result = {};
        if (Array.isArray(parsed.enabledManeuvers)) {
            result.enabledManeuvers = parsed.enabledManeuvers.filter(isManeuverId);
        }
        if (typeof parsed.enableHistoryExaminer === 'boolean') {
            result.enableHistoryExaminer = parsed.enableHistoryExaminer;
        }
        if (typeof parsed.enableInvestigations === 'boolean') {
            result.enableInvestigations = parsed.enableInvestigations;
        }
        if (Array.isArray(parsed.stageOrder)) {
            result.stageOrder = normalizeStageOrder(parsed.stageOrder.filter(isMainStage));
        }
        const openingMessages = parseManeuverOpeningMessages(parsed.maneuverOpeningMessages);
        if (Object.keys(openingMessages).length) {
            result.maneuverOpeningMessages = openingMessages;
        }
        const labels = parseManeuverLabels(parsed.maneuverLabels);
        if (Object.keys(labels).length) {
            result.maneuverLabels = labels;
        }
        if (parsed.patientBehavior && typeof parsed.patientBehavior === 'object') {
            const rawBehavior = parsed.patientBehavior;
            const partial = {};
            if (typeof rawBehavior.instructions === 'string') {
                partial.instructions = rawBehavior.instructions.trim();
            }
            if (typeof rawBehavior.tone === 'string') {
                partial.tone = rawBehavior.tone.trim();
            }
            if (typeof rawBehavior.emotion === 'string') {
                partial.emotion = rawBehavior.emotion.trim();
            }
            if (isPreferredLanguage(rawBehavior.preferredLanguage)) {
                partial.preferredLanguage = rawBehavior.preferredLanguage;
            }
            if (typeof rawBehavior.constraints === 'string') {
                partial.constraints = rawBehavior.constraints.trim();
            }
            if (Object.keys(partial).length) {
                result.patientBehavior = partial;
            }
        }
        return result;
    }
    catch {
        return {};
    }
}
export function mergeStationConfig(base, override) {
    if (!override) {
        return {
            ...base,
            enabledManeuvers: [...base.enabledManeuvers],
            stageOrder: [...base.stageOrder],
            maneuverOpeningMessages: { ...base.maneuverOpeningMessages },
            maneuverLabels: { ...base.maneuverLabels },
            patientBehavior: { ...base.patientBehavior },
        };
    }
    return {
        enabledManeuvers: override.enabledManeuvers?.length
            ? [...override.enabledManeuvers]
            : [...base.enabledManeuvers],
        enableHistoryExaminer: override.enableHistoryExaminer ?? base.enableHistoryExaminer,
        enableInvestigations: override.enableInvestigations ?? base.enableInvestigations,
        stageOrder: override.stageOrder?.length
            ? normalizeStageOrder(override.stageOrder)
            : [...base.stageOrder],
        maneuverOpeningMessages: mergeManeuverOpeningMessages(base.maneuverOpeningMessages, override.maneuverOpeningMessages),
        maneuverLabels: mergeManeuverLabels(base.maneuverLabels, override.maneuverLabels),
        patientBehavior: mergePatientBehavior(base.patientBehavior, override.patientBehavior),
    };
}
export function serializeStationConfig(config) {
    const enabled = config.enabledManeuvers.filter(isManeuverId);
    const openingMessages = parseManeuverOpeningMessages(config.maneuverOpeningMessages);
    const labels = parseManeuverLabels(config.maneuverLabels);
    const behavior = parsePatientBehavior(config.patientBehavior);
    return JSON.stringify({
        enabledManeuvers: enabled.length > 0 ? enabled : [...ALL_MANEUVERS],
        enableHistoryExaminer: config.enableHistoryExaminer !== false,
        enableInvestigations: config.enableInvestigations !== false,
        stageOrder: normalizeStageOrder(config.stageOrder),
        ...(Object.keys(openingMessages).length ? { maneuverOpeningMessages: openingMessages } : {}),
        ...(Object.keys(labels).length ? { maneuverLabels: labels } : {}),
        ...(patientBehaviorHasContent(behavior) ? { patientBehavior: behavior } : {}),
    });
}
export function serializePartialStationConfig(config) {
    const payload = {};
    if (config.enabledManeuvers?.length) {
        payload.enabledManeuvers = config.enabledManeuvers.filter(isManeuverId);
    }
    if (typeof config.enableHistoryExaminer === 'boolean') {
        payload.enableHistoryExaminer = config.enableHistoryExaminer;
    }
    if (typeof config.enableInvestigations === 'boolean') {
        payload.enableInvestigations = config.enableInvestigations;
    }
    if (config.stageOrder?.length) {
        payload.stageOrder = normalizeStageOrder(config.stageOrder.filter(isMainStage));
    }
    const openingMessages = parseManeuverOpeningMessages(config.maneuverOpeningMessages);
    if (Object.keys(openingMessages).length) {
        payload.maneuverOpeningMessages = openingMessages;
    }
    const labels = parseManeuverLabels(config.maneuverLabels);
    if (Object.keys(labels).length) {
        payload.maneuverLabels = labels;
    }
    if (config.patientBehavior) {
        const raw = config.patientBehavior;
        const partial = {};
        if (typeof raw.instructions === 'string')
            partial.instructions = raw.instructions.trim();
        if (typeof raw.tone === 'string')
            partial.tone = raw.tone.trim();
        if (typeof raw.emotion === 'string')
            partial.emotion = raw.emotion.trim();
        if (isPreferredLanguage(raw.preferredLanguage)) {
            partial.preferredLanguage = raw.preferredLanguage;
        }
        if (typeof raw.constraints === 'string')
            partial.constraints = raw.constraints.trim();
        if (Object.keys(partial).length) {
            payload.patientBehavior = partial;
        }
    }
    return JSON.stringify(payload);
}
export function getEnabledMainStages(config) {
    return config.stageOrder.filter((stage) => {
        if (stage === 'investigations')
            return config.enableInvestigations;
        return true;
    });
}
export function getSimulationStages(config) {
    return [...getEnabledMainStages(config), 'feedback'];
}
export function getNextMainStageAfter(current, config) {
    const stages = getEnabledMainStages(config);
    const index = stages.indexOf(current);
    if (index === -1)
        return stages[0] ?? 'diagnosis';
    return stages[index + 1] ?? 'feedback';
}
export function isManeuverEnabled(config, maneuverId) {
    return config.enabledManeuvers.includes(maneuverId);
}
export function getSessionStationConfig(session) {
    if (session.resolvedStationConfig?.trim()) {
        return parseStationConfig(session.resolvedStationConfig);
    }
    return parseStationConfig(session.case.stationConfig);
}
