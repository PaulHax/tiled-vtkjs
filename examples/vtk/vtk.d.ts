import "@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow";

// FullScreenRenderWindow accepts rootContainer at runtime; its published
// initialization type omits that DOM attachment option.
declare module "@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow" {
  interface IFullScreenRenderWindowInitialValues {
    rootContainer?: HTMLElement;
  }
}
