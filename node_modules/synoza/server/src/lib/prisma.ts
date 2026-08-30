import { PrismaClient } from '@prisma/client';
import { PrismaLibSql } from '@prisma/adapter-libsql';

const databaseUrl = process.env.DATABASE_URL || 'file:./dev.db';
const isTurso = databaseUrl.startsWith('libsql://') || databaseUrl.startsWith('https://');

function createPrismaClient(): PrismaClient {
  if (isTurso) {
    const adapter = new PrismaLibSql({
      url: databaseUrl,
      authToken: process.env.TURSO_AUTH_TOKEN,
    });
    return new PrismaClient({ adapter });
  }
  return new PrismaClient();
}

export const prisma = createPrismaClient();

