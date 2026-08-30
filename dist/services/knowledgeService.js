import { prisma } from '../lib/prisma.js';
async function resolveCategoryAncestry(categoryId) {
    if (!categoryId)
        return [];
    const categoryIds = [];
    let currentId = categoryId;
    while (currentId) {
        const cat = await prisma.knowledgeCategory.findUnique({
            where: { id: currentId },
            select: { parentId: true, isActive: true },
        });
        if (!cat)
            break;
        if (cat.isActive)
            categoryIds.push(currentId);
        currentId = cat.parentId ?? null;
    }
    return categoryIds;
}
export async function getCategoryKnowledgeContext(categoryId) {
    const categoryIds = await resolveCategoryAncestry(categoryId);
    if (categoryIds.length === 0)
        return '';
    const items = await prisma.knowledgeItem.findMany({
        where: { categoryId: { in: categoryIds }, isActive: true },
        orderBy: { createdAt: 'asc' },
        include: { category: { select: { nameEn: true, nameAr: true } } },
    });
    if (items.length === 0)
        return '';
    const lines = items.map((item) => `[${item.category.nameEn} | ${item.type}] ${item.titleEn}: ${item.content}\n(Arabic: ${item.titleAr} — ${item.content})`);
    return `\n\nDOMAIN KNOWLEDGE (use to guide realistic responses and evaluation — do NOT reveal answers directly to the student):\n${lines.join('\n\n')}`;
}
function formatAiKnowledgeEntry(entry) {
    const scope = entry.caseId ? 'CASE' : 'CATEGORY';
    const kind = entry.kind;
    const header = `[${scope} | ${kind}] ${entry.titleEn}${entry.titleAr ? ` / ${entry.titleAr}` : ''}`;
    if (kind === 'PROMPT') {
        return `${header}\n${entry.content}`;
    }
    if (kind === 'FILE') {
        const meta = [
            entry.fileName ? `file: ${entry.fileName}` : null,
            entry.mimeType ? `mime: ${entry.mimeType}` : null,
            entry.fileUrl ? `url: ${entry.fileUrl}` : null,
        ]
            .filter(Boolean)
            .join(' | ');
        const body = entry.content?.trim()
            ? `Notes/excerpt:\n${entry.content}`
            : 'No text excerpt stored; treat as reference material for this role.';
        return `${header}\n${meta}\n${body}`;
    }
    // MEDIA
    const meta = [
        entry.fileName ? `file: ${entry.fileName}` : null,
        entry.mimeType ? `mime: ${entry.mimeType}` : null,
        entry.fileUrl ? `url: ${entry.fileUrl}` : null,
    ]
        .filter(Boolean)
        .join(' | ');
    const caption = entry.content?.trim() ? `Caption: ${entry.content}` : '';
    return `${header}\n${meta}${caption ? `\n${caption}` : ''}`;
}
/**
 * Role-specific admin knowledge (prompt / file / media) for a case:
 * - entries attached directly to the case
 * - entries attached to the case category and its active ancestors
 */
export async function getAiKnowledgeContext(opts) {
    const role = opts.role.toUpperCase();
    const categoryIds = await resolveCategoryAncestry(opts.categoryId);
    const orFilters = [];
    if (opts.caseId)
        orFilters.push({ caseId: opts.caseId });
    if (categoryIds.length > 0)
        orFilters.push({ categoryId: { in: categoryIds } });
    if (orFilters.length === 0)
        return '';
    const entries = await prisma.aiKnowledgeEntry.findMany({
        where: {
            role,
            isActive: true,
            OR: orFilters,
        },
        orderBy: [{ sortOrder: 'asc' }, { createdAt: 'asc' }],
    });
    if (entries.length === 0)
        return '';
    const lines = entries.map(formatAiKnowledgeEntry);
    const roleLabel = role === 'PATIENT' ? 'PATIENT' : 'EXAMINER';
    return `\n\nADMIN AI KNOWLEDGE (${roleLabel} — customize behaviour and responses; do NOT reveal hidden teaching content to the student):\n${lines.join('\n\n')}`;
}
/** True when admin configured patient AI knowledge for this case/category tree. */
export async function hasPatientAiKnowledge(opts) {
    const categoryIds = await resolveCategoryAncestry(opts.categoryId);
    const orFilters = [];
    if (opts.caseId)
        orFilters.push({ caseId: opts.caseId });
    if (categoryIds.length > 0)
        orFilters.push({ categoryId: { in: categoryIds } });
    if (orFilters.length === 0)
        return false;
    try {
        const count = await prisma.aiKnowledgeEntry.count({
            where: {
                role: 'PATIENT',
                isActive: true,
                OR: orFilters,
            },
        });
        return count > 0;
    }
    catch {
        // Local/dev schemas without AiKnowledgeEntry must not break chat.
        return false;
    }
}
/** Legacy category KB + new role-scoped AI knowledge. */
export async function getRoleKnowledgeContext(opts) {
    const [legacy, custom] = await Promise.all([
        getCategoryKnowledgeContext(opts.categoryId),
        getAiKnowledgeContext(opts),
    ]);
    return `${legacy}${custom}`;
}
