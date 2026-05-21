/**
 * Cloudflare Worker - Stock Prediction API Proxy
 *
 * This worker proxies requests to the Python prediction backend.
 * Deploy with: wrangler deploy
 *
 * Environment variables:
 *   PREDICTION_API_URL - Backend URL (default: http://localhost:8000)
 *   API_SECRET         - Required secret key for external client authentication
 *   INTERNAL_API_SECRET - Optional secret key for internal backend authentication
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const backendUrl = env.PREDICTION_API_URL || "http://localhost:8000";
    const apiSecret = env.API_SECRET;
    const internalSecret = env.INTERNAL_API_SECRET;
    const internalHeaders = internalSecret ? { "X-Internal-Auth": internalSecret } : {};

    // CORS handling
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization",
          "Access-Control-Max-Age": "86400",
        },
      });
    }

    // Auth check - API_SECRET is required
    if (!apiSecret) {
      return new Response(JSON.stringify({ error: "API_SECRET not configured" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }

    const authHeader = request.headers.get("Authorization");
    if (!authHeader || authHeader !== `Bearer ${apiSecret}`) {
      return new Response(JSON.stringify({ error: "Unauthorized" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Route: /health
    if (url.pathname === "/health") {
      try {
        const resp = await fetch(`${backendUrl}/health`);
        const data = await resp.json();
        return jsonResponse(data, resp.status);
      } catch {
        return jsonResponse({ status: "error", message: "Backend unreachable" }, 503);
      }
    }

    // Route: /predict (GET or POST) → proxies to /predict/quick for speed
    if (url.pathname === "/predict" || url.pathname === "/predict/") {
      if (request.method === "POST") {
        try {
          const body = await request.json();
          const resp = await fetch(`${backendUrl}/predict/quick`, {
            method: "POST",
            headers: { "Content-Type": "application/json", ...internalHeaders },
            body: JSON.stringify(body),
          });
          const data = await resp.json();
          return jsonResponse(data, resp.status);
        } catch (e) {
          return jsonResponse({ error: `Invalid request: ${e.message}` }, 400);
        }
      }

      // GET - forward query params
      const params = url.searchParams;
      const query = new URLSearchParams();
      for (const [key, value] of params) {
        query.append(key, value);
      }

      try {
        const resp = await fetch(`${backendUrl}/predict/quick?${query.toString()}`, { headers: internalHeaders });
        const data = await resp.json();
        return jsonResponse(data, resp.status);
      } catch (e) {
        return jsonResponse({ error: `Prediction failed: ${e.message}` }, 500);
      }
    }

    // Route: /predict/full (GET) → full prediction with ML training
    if (url.pathname === "/predict/full" || url.pathname === "/predict/full/") {
      const params = url.searchParams;
      const query = new URLSearchParams();
      for (const [key, value] of params) {
        query.append(key, value);
      }

      try {
        const resp = await fetch(`${backendUrl}/predict?${query.toString()}`, { headers: internalHeaders });
        const data = await resp.json();
        return jsonResponse(data, resp.status);
      } catch (e) {
        return jsonResponse({ error: `Full prediction failed: ${e.message}` }, 500);
      }
    }

    // Route: /predict/cache (GET) → cached prediction
    if (url.pathname === "/predict/cache" || url.pathname === "/predict/cache/") {
      const params = url.searchParams;
      const query = new URLSearchParams();
      for (const [key, value] of params) {
        query.append(key, value);
      }

      try {
        const resp = await fetch(`${backendUrl}/predict/cache?${query.toString()}`, { headers: internalHeaders });
        const data = await resp.json();
        return jsonResponse(data, resp.status);
      } catch (e) {
        return jsonResponse({ error: `Cache lookup failed: ${e.message}` }, 500);
      }
    }

    // Route: /stocks/{code}/info
    if (url.pathname.startsWith("/stocks/")) {
      const code = url.pathname.split("/")[2];
      if (!code) {
        return jsonResponse({ error: "Stock code required" }, 400);
      }

      try {
        const resp = await fetch(`${backendUrl}/stocks/${code}/info`, { headers: internalHeaders });
        const data = await resp.json();
        return jsonResponse(data, resp.status);
      } catch (e) {
        return jsonResponse({ error: `Stock info failed: ${e.message}` }, 500);
      }
    }

    // Route: /stocks?zone=cn|hk|us
    if (url.pathname === "/stocks" || url.pathname === "/stocks/") {
      const zone = url.searchParams.get("zone") || "cn";
      try {
        const resp = await fetch(`${backendUrl}/stocks?zone=${zone}`, { headers: internalHeaders });
        const data = await resp.json();
        return jsonResponse(data, resp.status);
      } catch (e) {
        return jsonResponse({ error: `Stock list failed: ${e.message}` }, 500);
      }
    }

    // 404
    return jsonResponse({ error: "Not found" }, 404);
  },
};

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
    },
  });
}
