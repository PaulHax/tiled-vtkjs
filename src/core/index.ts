export type { Vec3, Bounds } from "./geometry";

export {
  cursorRay,
  type CameraView,
  type PerspectiveCameraView,
  type OrthographicCameraView,
  type CursorRay,
} from "./camera";

export { validateAffineMatrix, type Mat16, type Mat4 } from "./mat4";

export {
  createMemoryPool,
  defaultMemoryBudgetBytes,
  DEFAULT_MEMORY_BUDGET_BYTES,
  type MemoryPool,
  type MemoryPoolMember,
  type MemoryPoolOptions,
} from "./memoryPool";

export {
  createSubmissionScheduler,
  type SubmissionSchedulerOptions,
  type Submission,
  type SubmissionJob,
  type SubmissionScheduler,
  type SubmissionSchedulerStats,
} from "./submissionScheduler";

export {
  createStreamedMemberFactoryRegistry,
  type MemberFactoryRegistration,
  type StreamedMemberFactory,
  type StreamedMemberFactoryRegistry,
} from "./memberFactoryRegistry";

export type {
  Allocation,
  AllocationRegime,
  GovernorInputs,
  Importance,
  MemberPickResult,
  OcclusionResult,
  OutstandingWork,
  StreamedMember,
  StreamedMemberContext,
} from "./streamedMember";

export { scenePoint, type ScenePoint } from "./frames";

export { CULLED, importanceFromRootSseCssPx } from "./streamedMember";

export {
  createStreamedSceneCoordinator,
  type AdaptiveQualityTargets,
  type StreamedCoordinatorMemberStats,
  type StreamedMemberRegistration,
  type StreamedMemberRegistrationOptions,
  type StreamedSceneCoordinator,
  type StreamedSceneCoordinatorOptions,
  type StreamedSceneCoordinatorStats,
} from "./streamedSceneCoordinator";

export {
  ADAPTIVE_QUALITY_DEFAULTS,
  type AdaptiveQualityOptions,
  type QualityAdjustment,
  type QualityAdjustmentDirection,
  type QualityAdjustmentReason,
  type QualityRegime,
} from "./adaptiveBudget";

export {
  createViewGovernor,
  type CapacitySampleMetrics,
  type FrameVerdict,
  type GovernorWorkState,
  type HostFrameMetrics,
  type MotionReference,
  type MotionSourceKind,
  type TransientFrameMetrics,
  type ViewGovernor,
  type ViewGovernorOptions,
  type ViewGovernorStats,
} from "./viewGovernor";
