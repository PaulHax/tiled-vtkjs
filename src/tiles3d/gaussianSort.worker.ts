import {
  createGaussianDistanceSorter,
  type GaussianSortRange,
  type GaussianSortWorkerRequest,
  type GaussianSortWorkerResponse,
} from "./gaussianSort";

const workerScope = globalThis as unknown as {
  onmessage: ((event: MessageEvent<GaussianSortWorkerRequest>) => void) | null;
  postMessage(
    message: GaussianSortWorkerResponse,
    transfer?: Transferable[],
  ): void;
};
const sorter = createGaussianDistanceSorter();
let ranges: readonly GaussianSortRange[] = [];
let sceneRevision: number | null = null;

workerScope.onmessage = ({ data }) => {
  if (data.kind === "scene") {
    sceneRevision = data.sceneRevision;
    ranges = data.ranges;
    return;
  }
  try {
    if (data.sceneRevision !== sceneRevision) {
      throw new Error("Gaussian sort scene revision is obsolete");
    }
    const order = sorter.sort(ranges, data.cameraPosition);
    workerScope.postMessage({ kind: "order", jobId: data.jobId, order }, [
      order.buffer,
    ]);
  } catch (error) {
    workerScope.postMessage({
      kind: "error",
      jobId: data.jobId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
