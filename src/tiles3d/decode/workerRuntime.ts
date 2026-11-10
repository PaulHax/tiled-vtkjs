import { decodeTileContent } from "./decoder";
import {
  TileDecodeError,
  TileUnsupportedExtensionError,
  TileUnsupportedGaussianProfileError,
} from "./types";
import { buildTransferList } from "./transfer";
import { ResourcePolicyError } from "../../core/memoryResources";
import type {
  DecodeWorkerErrorMessage,
  DecodeWorkerRequestMessage,
  DecodeMemoryResponseMessage,
} from "./pool";

type ClassicWorkerScope = {
  onmessage: ((event: MessageEvent<unknown>) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
};

const errorMessage = (
  message: DecodeWorkerRequestMessage,
  error: unknown,
): DecodeWorkerErrorMessage => ({
  kind: "error",
  jobId: message.jobId,
  error:
    error instanceof ResourcePolicyError
      ? {
          name: error.name,
          message: error.message,
          domain: error.domain,
          requestedBytes: error.requestedBytes,
          limitBytes: error.limitBytes,
          retryable: error.retryable,
        }
      : error instanceof TileDecodeError
        ? {
            name: error.name,
            message: error.message,
            tileUri: error.tileUri,
            stage: error.stage,
            reason: error.reason,
            ...(error instanceof TileUnsupportedExtensionError
              ? { extension: error.extension }
              : {}),
            ...(error instanceof TileUnsupportedGaussianProfileError
              ? { property: error.property, value: error.value }
              : {}),
          }
        : {
            name: error instanceof Error ? error.name : "Error",
            message: error instanceof Error ? error.message : String(error),
          },
});

export const installDecodeWorker = (scope: ClassicWorkerScope): void => {
  let nextReservation = 1;
  const reservations = new Map<
    number,
    { resolve(): void; reject(error: unknown): void }
  >();
  scope.onmessage = (event) => {
    const value = event.data;
    if (
      value &&
      typeof value === "object" &&
      "kind" in value &&
      value.kind === "memory-response"
    ) {
      const response = value as DecodeMemoryResponseMessage;
      const pending = reservations.get(response.requestId);
      reservations.delete(response.requestId);
      if (response.error)
        pending?.reject(
          new ResourcePolicyError(
            response.error.domain,
            response.error.requestedBytes,
            response.error.limitBytes,
            response.error.retryable,
          ),
        );
      else pending?.resolve();
      return;
    }
    if (
      !value ||
      typeof value !== "object" ||
      !("kind" in value) ||
      value.kind !== "decode" ||
      !("jobId" in value) ||
      typeof value.jobId !== "number" ||
      !("request" in value)
    ) {
      return;
    }
    const message = value as DecodeWorkerRequestMessage;
    let claimedBytes = message.request.content.byteLength * 2;
    const reserveMemory = message.managedMemory
      ? async (bytes: number): Promise<void> => {
          claimedBytes += bytes;
          const requestId = nextReservation++;
          await new Promise<void>((resolve, reject) => {
            reservations.set(requestId, { resolve, reject });
            scope.postMessage({
              kind: "memory-request",
              jobId: message.jobId,
              requestId,
              bytes: claimedBytes,
            });
          });
        }
      : undefined;
    void decodeTileContent(message.request, { reserveMemory }).then(
      (result) => {
        scope.postMessage(
          {
            kind: "complete",
            jobId: message.jobId,
            result,
          },
          buildTransferList(result),
        );
      },
      (error: unknown) => scope.postMessage(errorMessage(message, error)),
    );
  };
};
