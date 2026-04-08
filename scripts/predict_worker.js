/**
 * Cloudflare Worker - Stock Prediction API Proxy
 *
 * This worker proxies requests to the Python prediction backend.
 * Deploy with: wrangler deploy
 *
 * Environment variables:
 *   PREDICTION_API_URL - Backend URL (default: http://localhost:8000)
 *   API_SECRET         - Optional secret key for authentication
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const backendUrl = env.PREDICTION_API_URL || "http://localhost:8000";
    const apiSecret = env.API_SECRET || null;

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

    // Auth check
    if (apiSecret) {
      const authHeader = request.headers.get("Authorization");
      if (!authHeader || authHeader !== `Bearer ${apiSecret}`) {
        return new Response(JSON.stringify({ error: "Unauthorized" }), {
          status: 401,
          headers: { "Content-Type": "application/json" },
        });
      }
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

    // Route: /predict (GET or POST)
    if (url.pathname === "/predict" || url.pathname === "/predict/") {
      if (request.method === "POST") {
        try {
          const body = await request.json();
          const resp = await fetch(`${backendUrl}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
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
        const resp = await fetch(`${backendUrl}/predict?${query.toString()}`);
        const data = await resp.json();
        return jsonResponse(data, resp.status);
      } catch (e) {
        return jsonResponse({ error: `Prediction failed: ${e.message}` }, 500);
      }
    }

    // Route: /stocks/{code}/info
    if (url.pathname.startsWith("/stocks/")) {
      const code = url.pathname.split("/")[2];
      if (!code) {
        return jsonResponse({ error: "Stock code required" }, 400);
      }

      try {
        const resp = await fetch(`${backendUrl}/stocks/${code}/info`);
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
        const resp = await fetch(`${backendUrl}/stocks?zone=${zone}`);
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
