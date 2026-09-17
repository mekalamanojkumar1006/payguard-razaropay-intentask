FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY frontend/package.json frontend/package-lock.json* /app/
RUN npm install

# Copy source
COPY frontend/ /app/

# Expose port
EXPOSE 3000

# Start development server on port 3000
CMD ["npm", "run", "dev", "--", "--port", "3000", "--host", "0.0.0.0"]
