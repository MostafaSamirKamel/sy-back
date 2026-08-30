FROM node:20-slim

# Install OpenSSL (required by Prisma) and FFmpeg for audio processing
RUN apt-get update -y && apt-get install -y openssl ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definitions
COPY package*.json ./
COPY prisma ./prisma/

# Install dependencies (fresh Linux build)
RUN npm install

# Copy source code
COPY tsconfig.json ./
COPY src ./src/
COPY scripts ./scripts/

# Generate Prisma Client & compile TypeScript
RUN npx prisma generate
RUN npm run build

ENV NODE_ENV=production
ENV PORT=5000
ENV DATABASE_URL="file:/app/prisma/dev.db"
ENV JWT_SECRET="synoza-jwt-default-secret-key-super-secure-2026"
EXPOSE 5000

CMD ["npm", "start"]
