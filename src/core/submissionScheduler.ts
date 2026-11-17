import { finiteAtLeast, wholeAtLeast } from "./numeric";
import { safeCall } from "./observers";

const MiB = 1024 * 1024;

/** Default main-thread/GPU admission limits per animation frame. */
export const DEFAULT_SUBMISSION_BYTES_PER_FRAME = 16 * MiB;
export const DEFAULT_SUBMISSION_TIME_MS_PER_FRAME = 4;

export type SubmissionJob = {
  /** Managed preparation or transfer bytes processed by this job. */
  readonly bytes: number;
  /** Indivisible resource already checked against its owner's memory budget.
   * If larger than the byte slice, admit it alone; all other work waits. */
  readonly atomic?: boolean;
  /** Publish visible state only when a paint will follow. */
  readonly requiresPaint?: boolean;
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
  readonly schedulePreparation?: () => void;
  readonly maxBytesPerFrame?: number;
  readonly maxTimeMsPerFrame?: number;
  /** Injectable monotonic clock for deterministic tests. */
  readonly now?: () => number;
  readonly budget?: SubmissionFrameBudget;
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
  /** Drain one bounded slice, leaving publication for a paint if needed. */
  prepareFrame(pageFrame?: number, painting?: boolean): void;
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

type SubmissionFrameBudget = ReturnType<typeof createSubmissionFrameBudget>;

export const createSubmissionFrameBudget = (
  maxBytes = DEFAULT_SUBMISSION_BYTES_PER_FRAME,
  maxTimeMs = DEFAULT_SUBMISSION_TIME_MS_PER_FRAME,
) => {
  const deferred = new Set<object>();
  const lastGrant = new WeakMap<object, number>();
  let grants = 0;
  let priority: object | undefined;
  let frame: number | undefined;
  let bytes = 0;
  let elapsed = 0;
  let jobs = 0;

  return {
    maxBytes,
    maxTimeMs,
    begin(pageFrame: number) {
      if (frame === pageFrame) return;
      frame = pageFrame;
      bytes = elapsed = jobs = 0;
      // A view that ran out of allowance is first in line the next time it
      // paints. A view that stops painting can hold one frame at most.
      priority = undefined;
      for (const view of deferred) {
        if (
          !priority ||
          (lastGrant.get(view) ?? 0) < (lastGrant.get(priority) ?? 0)
        )
          priority = view;
      }
      deferred.clear();
    },
    admit(view: object, size: number) {
      if (
        (priority && priority !== view) ||
        (jobs > 0 && (bytes + size > maxBytes || elapsed >= maxTimeMs))
      ) {
        deferred.add(view);
        return false;
      }
      if (priority === view) priority = undefined;
      lastGrant.set(view, ++grants);
      deferred.delete(view);
      bytes += size;
      jobs += 1;
      return true;
    },
    spend(ms: number) {
      elapsed += ms;
    },
    withdraw(view: object) {
      deferred.delete(view);
      if (priority === view) priority = undefined;
    },
  };
};

export const createSubmissionScheduler = (
  options: SubmissionSchedulerOptions,
): SubmissionScheduler => {
  const maxBytes = wholeAtLeast(
    "maxBytesPerFrame",
    options.budget?.maxBytes ??
      options.maxBytesPerFrame ??
      DEFAULT_SUBMISSION_BYTES_PER_FRAME,
    1,
  );
  const maxTimeMs = finiteAtLeast(
    "maxTimeMsPerFrame",
    options.budget?.maxTimeMs ??
      options.maxTimeMsPerFrame ??
      DEFAULT_SUBMISSION_TIME_MS_PER_FRAME,
    0,
  );
  const now = options.now ?? (() => performance.now());
  const budget =
    options.budget ?? createSubmissionFrameBudget(maxBytes, maxTimeMs);
  const view = {};
  let ownFrame = 0;
  const queue: QueuedJob[] = [];
  let queuedBytes = 0;
  let lastFrameAdmittedJobs = 0;
  let lastFrameAdmittedBytes = 0;
  let lastFrameElapsedMs = 0;
  let peakQueuedJobs = 0;
  let peakFrameAdmittedBytes = 0;
  let disposed = false;
  let draining = false;

  const discardCancelledHead = (): void => {
    while (queue.length && !queue[0]!.job) queue.shift();
  };

  const requestHead = (): void => {
    if (queue[0]?.job?.requiresPaint || !options.schedulePreparation)
      options.scheduleRender();
    else options.schedulePreparation();
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
        if (wasEmpty && !draining) requestHead();
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
          if (!queue.length) budget.withdraw(view);
          return true;
        },
      };
    },

    prepareFrame(pageFrame = ++ownFrame, painting = true) {
      lastFrameAdmittedJobs = 0;
      lastFrameAdmittedBytes = 0;
      lastFrameElapsedMs = 0;
      if (disposed) return;
      budget.begin(pageFrame);
      let elapsed = 0;
      let frameBytes = 0;
      let frameJobs = 0;
      discardCancelledHead();
      draining = true;
      while (queue.length > 0) {
        const next = queue[0]!;
        if (
          (!painting && next.job!.requiresPaint) ||
          !budget.admit(view, next.bytes)
        )
          break;
        queue.shift();
        next.settled = true;
        queuedBytes -= next.bytes;
        const job = next.job!;
        next.job = null;
        const started = now();
        try {
          job.run();
        } catch (error) {
          safeCall(() => job.onError?.(error));
        }
        const spent = Math.max(0, now() - started);
        budget.spend(spent);
        elapsed += spent;
        frameJobs += 1;
        frameBytes += next.bytes;
        discardCancelledHead();
      }
      draining = false;
      lastFrameAdmittedJobs = frameJobs;
      lastFrameAdmittedBytes = frameBytes;
      lastFrameElapsedMs = elapsed;
      peakFrameAdmittedBytes = Math.max(peakFrameAdmittedBytes, frameBytes);
      if (queue.length > 0) requestHead();
      else budget.withdraw(view);
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
      budget.withdraw(view);
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
  steps: Iterable<Pick<SubmissionJob, "bytes" | "run" | "requiresPaint">>,
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
        requiresPaint: step.value.requiresPaint,
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
