import { finiteAtLeast, wholeAtLeast } from "./numeric";
import { safeCall } from "./observers";

const MiB = 1024 * 1024;

/** Default main-thread/GPU admission limits applied before each paint. */
export const DEFAULT_SUBMISSION_BYTES_PER_FRAME = 16 * MiB;
export const DEFAULT_SUBMISSION_TIME_MS_PER_FRAME = 4;

export type SubmissionJob = {
  /** Managed preparation or transfer bytes processed by this job. */
  readonly bytes: number;
  /** Indivisible resource already checked against its owner's memory budget.
   * If larger than the byte slice, admit it alone; all other work waits. */
  readonly atomic?: boolean;
  /** Synchronous VTK construction or GPU-admission work. */
  readonly run: () => void;
  readonly onError?: (error: unknown) => void;
};

export type Submission = {
  /** Cancel while queued. Returns true only for the first successful cancel. */
  cancel(): boolean;
};

export type SubmissionSchedulerOptions = {
  readonly scheduleRender: () => void;
  readonly maxBytesPerFrame?: number;
  readonly maxTimeMsPerFrame?: number;
  /** Injectable monotonic clock for deterministic tests. */
  readonly now?: () => number;
};

export type SubmissionSchedulerStats = {
  readonly maxBytesPerFrame: number;
  readonly maxTimeMsPerFrame: number;
  readonly queuedJobs: number;
  readonly queuedBytes: number;
  readonly lastFrameAdmittedJobs: number;
  readonly lastFrameAdmittedBytes: number;
  readonly lastFrameElapsedMs: number;
  readonly peakQueuedJobs: number;
  readonly peakFrameAdmittedBytes: number;
};

export type SubmissionScheduler = {
  enqueue(job: SubmissionJob): Submission;
  /** Drain one bounded admission slice immediately before a view paint. */
  prepareFrame(): void;
  hasPending(): boolean;
  stats(): SubmissionSchedulerStats;
  dispose(): void;
};

/** `settled` means admitted or dropped, so it can no longer be cancelled. */
type QueuedJob = {
  bytes: number;
  job: SubmissionJob | null;
  settled: boolean;
};

export const createSubmissionScheduler = (
  options: SubmissionSchedulerOptions,
): SubmissionScheduler => {
  const maxBytes = wholeAtLeast(
    "maxBytesPerFrame",
    options.maxBytesPerFrame ?? DEFAULT_SUBMISSION_BYTES_PER_FRAME,
    1,
  );
  const maxTimeMs = finiteAtLeast(
    "maxTimeMsPerFrame",
    options.maxTimeMsPerFrame ?? DEFAULT_SUBMISSION_TIME_MS_PER_FRAME,
    0,
  );
  const now = options.now ?? (() => performance.now());
  const queue: QueuedJob[] = [];
  let queuedBytes = 0;
  let lastFrameAdmittedJobs = 0;
  let lastFrameAdmittedBytes = 0;
  let lastFrameElapsedMs = 0;
  let peakQueuedJobs = 0;
  let peakFrameAdmittedBytes = 0;
  let disposed = false;

  const discardCancelledHead = (): void => {
    while (queue.length && !queue[0]!.job) queue.shift();
  };

  return {
    enqueue(job) {
      const bytes = wholeAtLeast("submission bytes", job.bytes, 0);
      if (bytes > maxBytes && !job.atomic) {
        throw new Error(
          `submission job is ${bytes} bytes, exceeding the ${maxBytes}-byte per-frame cap; split it into smaller jobs`,
        );
      }
      const queued: QueuedJob = {
        job,
        bytes,
        settled: false,
      };
      if (!disposed) {
        const wasEmpty = queue.length === 0;
        queue.push(queued);
        queuedBytes += bytes;
        peakQueuedJobs = Math.max(peakQueuedJobs, queue.length);
        if (wasEmpty) options.scheduleRender();
      } else {
        queued.job = null;
        queued.settled = true;
      }
      return {
        cancel() {
          if (!queued.job || queued.settled) return false;
          queued.job = null;
          queuedBytes -= queued.bytes;
          discardCancelledHead();
          return true;
        },
      };
    },

    prepareFrame() {
      lastFrameAdmittedJobs = 0;
      lastFrameAdmittedBytes = 0;
      lastFrameElapsedMs = 0;
      if (disposed) return;
      const started = now();
      let frameBytes = 0;
      let frameJobs = 0;
      discardCancelledHead();
      while (queue.length > 0) {
        const next = queue[0]!;
        // An indivisible resource may consume a whole frame on its own.
        // Never combine an oversized atomic job with another job.
        if (
          frameJobs > 0 &&
          (frameBytes + next.bytes > maxBytes || now() - started >= maxTimeMs)
        ) {
          break;
        }
        queue.shift();
        next.settled = true;
        queuedBytes -= next.bytes;
        const job = next.job!;
        next.job = null;
        try {
          job.run();
        } catch (error) {
          safeCall(() => job.onError?.(error));
        }
        frameJobs += 1;
        frameBytes += next.bytes;
        discardCancelledHead();
      }
      lastFrameAdmittedJobs = frameJobs;
      lastFrameAdmittedBytes = frameBytes;
      lastFrameElapsedMs = Math.max(0, now() - started);
      peakFrameAdmittedBytes = Math.max(peakFrameAdmittedBytes, frameBytes);
      if (queue.length > 0) options.scheduleRender();
    },

    hasPending: () => !disposed && queue.some((entry) => entry.job),

    stats: () => ({
      maxBytesPerFrame: maxBytes,
      maxTimeMsPerFrame: maxTimeMs,
      queuedJobs: queue.reduce(
        (count, entry) => count + (entry.job ? 1 : 0),
        0,
      ),
      queuedBytes,
      lastFrameAdmittedJobs,
      lastFrameAdmittedBytes,
      lastFrameElapsedMs,
      peakQueuedJobs,
      peakFrameAdmittedBytes,
    }),

    dispose() {
      if (disposed) return;
      disposed = true;
      for (const entry of queue) {
        entry.job = null;
        entry.settled = true;
      }
      queue.length = 0;
      queuedBytes = 0;
    },
  };
};

/** Keep one continuation in the shared admission queue per preparation. */
export const enqueueSubmissionSteps = (
  scheduler: SubmissionScheduler,
  steps: Iterable<Pick<SubmissionJob, "bytes" | "run">>,
  onError: (error: unknown) => void,
  onProgress: () => void,
): Submission => {
  const iterator = steps[Symbol.iterator]();
  const frameCap = scheduler.stats().maxBytesPerFrame;
  let pending: Submission | undefined;
  let cancelled = false;
  const advance = (): void => {
    if (cancelled) return;
    const step = iterator.next();
    if (!step.done) {
      pending = scheduler.enqueue({
        bytes: step.value.bytes,
        atomic: step.value.bytes > frameCap,
        run() {
          step.value.run();
          onProgress();
          advance();
        },
        onError,
      });
    } else pending = undefined;
  };
  advance();
  return {
    cancel() {
      if (!pending || cancelled) return false;
      cancelled = true;
      pending.cancel();
      iterator.return?.();
      return true;
    },
  };
};
