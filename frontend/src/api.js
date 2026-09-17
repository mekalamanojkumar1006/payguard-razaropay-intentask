/**
 * PayGuard API Client.
 * Centralizes all backend calls.  Uses VITE_API_URL env var.
 */

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export async function getHealth() {
  return request("/health");
}

export async function getStats() {
  return request("/api/v1/stats");
}

export async function getTransactions(limit = 50, offset = 0) {
  return request(`/api/v1/transactions?limit=${limit}&offset=${offset}`);
}

export async function analyzeRisk(transaction) {
  return request("/api/v1/risk/analyze", {
    method: "POST",
    body: JSON.stringify(transaction),
  });
}

export async function investigateRisk(transaction) {
  return request("/api/v1/risk/investigate", {
    method: "POST",
    body: JSON.stringify(transaction),
  });
}
