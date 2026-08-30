import { Router } from 'express';
import { prisma } from '../lib/prisma.js';
const router = Router();
function buildTree(categories) {
    const map = new Map();
    const roots = [];
    for (const cat of categories) {
        map.set(cat.id, { ...cat, children: [] });
    }
    for (const cat of categories) {
        const node = map.get(cat.id);
        if (cat.parentId && map.has(cat.parentId)) {
            map.get(cat.parentId).children.push(node);
        }
        else {
            roots.push(node);
        }
    }
    const sortNodes = (nodes) => {
        nodes.sort((a, b) => a.sortOrder - b.sortOrder || a.nameEn.localeCompare(b.nameEn));
        nodes.forEach((n) => sortNodes(n.children));
    };
    sortNodes(roots);
    return roots;
}
router.get('/', async (_req, res) => {
    const categories = await prisma.knowledgeCategory.findMany({
        where: { isActive: true },
        orderBy: [{ sortOrder: 'asc' }, { nameEn: 'asc' }],
        include: {
            _count: { select: { cases: { where: { isPublished: true } }, children: { where: { isActive: true } } } },
        },
    });
    res.json({ categories: buildTree(categories) });
});
router.get('/:id', async (req, res) => {
    const category = await prisma.knowledgeCategory.findFirst({
        where: { id: req.params.id, isActive: true },
        include: {
            parent: { select: { id: true, nameEn: true, nameAr: true, parentId: true } },
            children: {
                where: { isActive: true },
                orderBy: [{ sortOrder: 'asc' }, { nameEn: 'asc' }],
                include: {
                    _count: { select: { cases: { where: { isPublished: true } }, children: { where: { isActive: true } } } },
                },
            },
            _count: { select: { cases: { where: { isPublished: true } } } },
        },
    });
    if (!category)
        return res.status(404).json({ error: 'Category not found' });
    res.json({ category });
});
export default router;
