import { installDecodeWorker } from "./decode/workerRuntime";

declare const self: Worker;

installDecodeWorker(self);
