const upstream =
  "https://3d-tiles-rendererjs-3dgs-plugin.vercel.app/api/blob/JastrebiaGora/";
const route =
  /^\/gaussian-city\/(tileset\.json|build_summary\.json|tiles\/\d+\/\d+\.glb)$/;

export const config = { path: "/gaussian-city/*" };

export default async function gaussianCity(request) {
  const match = route.exec(new URL(request.url).pathname);
  if (!match) return new Response(null, { status: 404 });
  if (request.method !== "GET" && request.method !== "HEAD")
    return new Response(null, { status: 405, headers: { Allow: "GET, HEAD" } });

  const range = request.headers.get("Range");
  try {
    const response = await fetch(new URL(match[1], upstream), {
      method: request.method,
      headers: range === null ? undefined : { Range: range },
      redirect: "follow",
      signal: request.signal,
    });
    const headers = new Headers();
    for (const name of [
      "Content-Type",
      "Content-Range",
      "Accept-Ranges",
      "Cache-Control",
    ]) {
      const value = response.headers.get(name);
      if (value !== null) headers.set(name, value);
    }
    return new Response(response.body, { status: response.status, headers });
  } catch {
    return new Response("Gaussian tileset host is unavailable", {
      status: 502,
    });
  }
}
