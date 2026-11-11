import { createGaussianSortWorkerBackend } from "../../../src/tiles3d";
import type { StreamedMemberRegistration } from "../../../src";
import {
  createGaussianAlphaPass,
  createGaussianContentAdapterFactory,
  createTiles3dMember,
} from "../../../src/tiles3d/vtk";
import type {
  Tiles3dMemberConfig,
  Tiles3dMemberStats,
} from "../../../src/tiles3d/memberTypes";
import { parseTileset } from "../../../src/tiles3d/tilesetSource";
import { createSceneHost } from "../scene/host";
import { decodeWasmUrls } from "../scene/decodeAssets";
import { DATASETS } from "./datasets";
import { tilesetPlacement } from "./placement";

const element = <T extends HTMLElement>(id: string): T => {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing ${id}`);
  return found as T;
};
const params = new URLSearchParams(location.search);
const numberOption = (
  key: string,
  fallback: number,
  minimum: number,
  maximum: number,
): number => {
  const input = Number(params.get(key) ?? fallback);
  return Number.isFinite(input)
    ? Math.min(maximum, Math.max(minimum, input))
    : fallback;
};
const MiB = 1024 * 1024;
const budget = Math.floor(numberOption("memory", 384, 64, 2048)) * MiB;
const maximumVisibleSplats = Math.floor(
  numberOption("splats", 2_000_000, 1000, 8_000_000),
);
const viewer = element<HTMLDivElement>("viewer");
const host = createSceneHost(viewer, { memoryBudgetBytes: budget });
host.setTranslucentPass(
  createGaussianAlphaPass(() => {
    const context = host.glContext();
    if (!context) throw new Error("WebGL2 is unavailable");
    return context;
  }),
);
const realization = createGaussianContentAdapterFactory({
  createSortBackend: () =>
    createGaussianSortWorkerBackend(
      new Worker(
        new URL("../../../src/tiles3d/gaussianSort.worker.ts", import.meta.url),
        { type: "module" },
      ),
    ),
  maximumVisibleSplats,
  resourceBudgetBytes: budget,
});

const select = element<HTMLSelectElement>("dataset");
const urlInput = element<HTMLInputElement>("tileset-url");
const status = element<HTMLParagraphElement>("status");
const errorElement = element<HTMLParagraphElement>("error");
const statsElement = element<HTMLDListElement>("stats");
const quality = element<HTMLInputElement>("quality");
const visibility = element<HTMLInputElement>("visible");
const format = new Intl.NumberFormat("en-US");
const megabytes = (bytes: number): string => `${(bytes / MiB).toFixed(1)} MiB`;
let generation = 0;
let controller: AbortController | null = null;
let release: (() => void) | null = null;
let member: ReturnType<typeof createTiles3dMember> | null = null;
let activeRegistration: StreamedMemberRegistration | null = null;
let config: Tiles3dMemberConfig | null = null;
let frameScene: (() => void) | null = null;
let totalTiles = 0;
let transferredBytes = 0;
let fetchedTiles = 0;
let disposed = false;

const setStatus = (state: string, message: string): void => {
  status.dataset.state = state;
  status.textContent = message;
};
const showError = (error: unknown): void => {
  errorElement.hidden = false;
  errorElement.textContent =
    error instanceof Error ? error.message : String(error);
};
const updateStats = (): void => {
  if (!member || disposed) return;
  const tiles = member.stats() as Tiles3dMemberStats;
  const splats = realization.stats(host.renderer);
  const values: [string, string][] = [
    ["Drawn splats", format.format(splats?.displayedSplats ?? 0)],
    [
      "Drawn tiles",
      `${format.format(tiles.renderer.drawnTiles)} / ${format.format(totalTiles)}`,
    ],
    ["Fetched tiles", format.format(fetchedTiles)],
    ["Transferred", megabytes(transferredBytes)],
    ["Resident resources", megabytes(splats?.residentBytes ?? 0)],
    ["Decoded cache", megabytes(tiles.queue?.decodedBytes ?? 0)],
    ["Resource allowance", megabytes(budget)],
    ["Visible splat limit", format.format(maximumVisibleSplats)],
    [
      "Sort",
      splats?.pendingSort
        ? "Working"
        : `${(splats?.sortMs ?? 0).toFixed(1)} ms`,
    ],
  ];
  statsElement.replaceChildren(
    ...values.flatMap(([label, value]) => {
      const term = document.createElement("dt");
      term.textContent = label;
      const definition = document.createElement("dd");
      definition.textContent = value;
      definition.dataset.stat = label.toLowerCase().replaceAll(" ", "-");
      return [term, definition];
    }),
  );
  if ((splats?.displayedSplats ?? 0) > 0) {
    setStatus(
      "ready",
      tiles.irreducibleBudget
        ? "Showing available detail. Increase the resource allowance for finer tiles."
        : tiles.queue?.workPending || splats?.pendingSort
          ? "Scene visible. Refining nearby tiles…"
          : "Scene ready. Move closer to explore finer detail.",
    );
  } else if (tiles.irreducibleBudget) {
    setStatus("error", "The coarsest tile exceeds the resource allowance.");
  }
};
host.onFrame(updateStats);
host.onBeforeFrame((view, devicePixelRatio) => {
  activeRegistration?.setCamera(view);
  activeRegistration?.setDevicePixelRatio(devicePixelRatio);
});

for (const dataset of DATASETS) {
  const option = document.createElement("option");
  option.value = dataset.id;
  option.textContent = dataset.label;
  select.append(option);
}
const selectDataset = (id: string): void => {
  const dataset = DATASETS.find((entry) => entry.id === id);
  select.value = dataset?.id ?? "custom";
  if (dataset) urlInput.value = dataset.url;
  const attribution = element<HTMLParagraphElement>("attribution");
  attribution.replaceChildren();
  if (dataset) {
    attribution.append(document.createTextNode(`${dataset.attribution}. `));
    const source = document.createElement("a");
    source.href = dataset.source;
    source.textContent = "Source scene";
    source.target = "_blank";
    source.rel = "noopener";
    const license = document.createElement("a");
    license.href = dataset.license;
    license.textContent = "CC BY 4.0";
    license.target = "_blank";
    license.rel = "noopener";
    attribution.append(source, document.createTextNode(" · "), license);
  }
};

const load = async (urlValue: string): Promise<void> => {
  const current = ++generation;
  controller?.abort();
  release?.();
  release = null;
  member = null;
  activeRegistration = null;
  realization.dispose(host.renderer);
  controller = new AbortController();
  const signal = controller.signal;
  transferredBytes = 0;
  fetchedTiles = 0;
  totalTiles = 0;
  errorElement.hidden = true;
  statsElement.replaceChildren();
  frameScene = null;
  setStatus("loading", "Reading the tile hierarchy…");
  try {
    const url = new URL(urlValue, location.href);
    if (!["http:", "https:"].includes(url.protocol))
      throw new Error("Use an HTTP or HTTPS tileset URL.");
    if (url.search || url.hash)
      throw new Error("Use a stable tileset URL without query parameters.");
    const requestUrl = DATASETS.some((entry) => entry.url === url.href)
      ? new URL("/gaussian-city/tileset.json", location.href)
      : url;
    const endpoint = new URL(".", requestUrl).href.replace(/\/$/, "");
    const response = await fetch(requestUrl, { signal });
    if (!response.ok)
      throw new Error(`Tileset request failed: HTTP ${response.status}`);
    const text = await response.text();
    const document = JSON.parse(text) as unknown;
    const source = parseTileset(document, endpoint);
    if (current !== generation || disposed) return;
    const placement = tilesetPlacement(
      source.root,
      DATASETS.some((entry) => entry.url === url.href) ||
        params.get("ecef") === "1",
    );
    totalTiles = source.tileById.size;
    transferredBytes = new TextEncoder().encode(text).byteLength;
    const locationText = placement.location
      ? `Tileset ECEF placement: ${placement.location[1].toFixed(5)}°, ${placement.location[0].toFixed(5)}°.`
      : "Local reconstruction coordinates. No geographic placement supplied.";
    element<HTMLParagraphElement>("location").textContent = locationText;
    config = {
      endpoint,
      revision: `dataset-${current}`,
      tilesetToScene: placement.tilesetToScene,
      maximumScreenSpaceErrorPx: Number(quality.value),
      cacheBytes: Math.floor(budget / 4),
      concurrency: 2,
      wasm: decodeWasmUrls(),
      fetchTileset: async () => ({
        ok: true,
        status: 200,
        statusText: "OK",
        json: async () => document,
      }),
      fetchContent: async (contentUrl, init) => {
        const result = await fetch(contentUrl, init);
        return {
          ok: result.ok,
          status: result.status,
          statusText: result.statusText,
          arrayBuffer: async () => {
            const bytes = await result.arrayBuffer();
            if (current === generation && !disposed) {
              transferredBytes += bytes.byteLength;
              fetchedTiles += 1;
            }
            return bytes;
          },
        };
      },
      onError: (error) => {
        if (current === generation && !disposed) showError(error);
      },
    };
    member = createTiles3dMember(
      host.coordinator.context(host.renderer),
      config,
      {
        ...host.tiles3dServices,
        createContentAdapter: realization,
      },
    );
    const registration = host.coordinator.register(member, {
      id: "gaussian-tiles",
      qualityManaged: true,
    });
    activeRegistration = registration;
    release = () => registration.release();
    registration.setActive(visibility.checked);
    frameScene = () => {
      host.lookAt(placement.center, placement.radius * 2.8, placement.bounds);
      registration.setCamera(host.cameraView());
    };
    frameScene();
    setStatus("loading", "Loading the first visible tiles…");
    const next = new URL(location.href);
    next.searchParams.set("tileset", url.href);
    history.replaceState(null, "", next);
    host.scheduleRender();
  } catch (error) {
    if (current !== generation || signal.aborted || disposed) return;
    setStatus("error", "Unable to open this tileset.");
    showError(error);
  }
};

element<HTMLFormElement>("dataset-form").addEventListener("submit", (event) => {
  event.preventDefault();
  void load(urlInput.value);
});
select.addEventListener("change", () => {
  selectDataset(select.value);
  if (select.value !== "custom") void load(urlInput.value);
});
element<HTMLButtonElement>("reset-view").addEventListener("click", () =>
  frameScene?.(),
);
quality.value = String(numberOption("sse", 24, 2, 96));
const updateQuality = (): void => {
  element<HTMLOutputElement>("quality-value").textContent =
    `${quality.value} px`;
  if (config && member) {
    config = { ...config, maximumScreenSpaceErrorPx: Number(quality.value) };
    activeRegistration?.setConfig(config);
    host.scheduleRender();
  }
};
quality.addEventListener("input", updateQuality);
updateQuality();
visibility.addEventListener("change", () => {
  activeRegistration?.setActive(visibility.checked);
  if (!visibility.checked) setStatus("hidden", "Scene hidden.");
  host.scheduleRender();
});

const dispose = (): void => {
  if (disposed) return;
  disposed = true;
  generation += 1;
  controller?.abort();
  release?.();
  realization.dispose(host.renderer);
  host.dispose();
  setStatus("disposed", "View closed.");
};
window.addEventListener("pagehide", dispose, { once: true });

if (params.has("test")) {
  Object.assign(window, {
    gaussianTiles: {
      stats: () => ({
        member: member?.stats(),
        renderer: realization.stats(host.renderer),
        fetchedTiles,
        transferredBytes,
        totalTiles,
        rendererName: host.rendererName,
      }),
      frameScene: () => frameScene?.(),
      camera: () => ({
        position: [...host.camera.getPosition()],
        target: [...host.camera.getFocalPoint()],
      }),
      moveCamera: (
        position: [number, number, number],
        target: [number, number, number],
      ) => {
        host.camera.setPosition(...position);
        host.camera.setFocalPoint(...target);
        host.scheduleRender();
      },
      dispose,
    },
  });
}

const initialUrl = params.get("tileset");
selectDataset(
  initialUrl
    ? (DATASETS.find((entry) => entry.url === initialUrl)?.id ?? "custom")
    : DATASETS[0]!.id,
);
if (initialUrl) urlInput.value = initialUrl;
void load(urlInput.value);
