FROM node:20-alpine

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install

# Build arg for VITE_API_URL
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}

COPY . ./

EXPOSE 5173
CMD ["npm", "run", "dev"]
