/** Pending source preparation belongs to its target dataset and view. */
export const createDatasetOperations = () => {
  const pending = new Map<object, AbortController>();
  let closed = false;
  const cancel = (key: object): void => {
    pending.get(key)?.abort();
    pending.delete(key);
  };
  return {
    cancel,
    begin(target: object | null) {
      const key = target ?? {};
      cancel(key);
      const controller = new AbortController();
      if (closed) controller.abort();
      else pending.set(key, controller);
      return {
        signal: controller.signal,
        check: () => controller.signal.throwIfAborted(),
        finish() {
          if (pending.get(key) === controller) pending.delete(key);
        },
      };
    },
    dispose() {
      closed = true;
      for (const key of pending.keys()) cancel(key);
    },
  };
};
