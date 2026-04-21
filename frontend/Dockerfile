# syntax=docker/dockerfile:1.7
# ---- Builder ----
FROM node:20-alpine AS builder
WORKDIR /app

# Install deps with a clean, deterministic install
COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

# Copy source and build the production bundle
COPY tsconfig*.json angular.json tailwind.config.js postcss.config.js ./
COPY src ./src

ENV NODE_ENV=production
RUN npm run build -- --configuration production

# ---- Runtime: nginx serving static bundle ----
FROM nginx:1.27-alpine AS runtime

# Drop default config and provide our SPA-aware one
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/uz-ews.conf

# Copy build output
COPY --from=builder /app/dist/student-dropout-frontend/browser /usr/share/nginx/html

# Run as non-root for safety
RUN addgroup -S app && adduser -S -G app app \
  && chown -R app:app /usr/share/nginx/html \
  && chown -R app:app /var/cache/nginx /var/log/nginx /etc/nginx/conf.d \
  && touch /var/run/nginx.pid \
  && chown -R app:app /var/run/nginx.pid

USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD wget --quiet --tries=1 --spider http://localhost:8080/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]
