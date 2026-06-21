# syntax=docker/dockerfile:1
# Frontend TimeCast. Контекст сборки = ../frontend.
# Стадии: deps -> dev (target: dev) | build -> prod (nginx).

# ---- deps ----
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---- dev (для docker-compose.dev: target: dev; исходники монтируются томом) ----
FROM node:20-alpine AS dev
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV CHOKIDAR_USEPOLLING=true
EXPOSE 3000
CMD ["npm", "start"]

# ---- build ----
FROM deps AS build
COPY . .
RUN npm run build

# ---- prod (nginx отдаёт собранную статику) ----
FROM nginx:alpine AS prod
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
