import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkPointGaussianMapper from "@kitware/vtk.js/Rendering/Core/PointGaussianMapper";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkPlaneSource from "@kitware/vtk.js/Filters/Sources/PlaneSource";
import vtkRenderer from "@kitware/vtk.js/Rendering/Core/Renderer";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkForwardPass from "@kitware/vtk.js/Rendering/OpenGL/ForwardPass";
import { createSubmissionScheduler } from "../../src/core/submissionScheduler";
import {
  createGaussianAlphaPass,
  createGaussianContentAdapterFactory,
} from "../../src/tiles3d/gaussianAdapter";

const generic = vtkGenericRenderWindow.newInstance({ background: [0, 0, 0] });
generic.setContainer(document.querySelector("#view"));
generic.resize();
const left = generic.getRenderer();
left.setViewport(0, 0, 0.5, 1);
const right = vtkRenderer.newInstance({ background: [0, 0, 0] });
right.setViewport(0.5, 0, 1, 1);
const renderWindow = generic.getRenderWindow();
renderWindow.addRenderer(right);
for (const renderer of [left, right]) {
  const camera = renderer.getActiveCamera();
  camera.setPosition(0, 0, 10);
  camera.setFocalPoint(0, 0, 0);
  camera.setParallelProjection(true);
  camera.setParallelScale(1);
  camera.setClippingRange(0.1, 100);
}
const openGL = generic.getApiSpecificRenderWindow();
renderWindow.render();
const gl = openGL.getContext();
openGL.setRenderPasses([
  vtkForwardPass.newInstance({
    translucentPass: createGaussianAlphaPass(() => openGL.getContext()),
  }),
]);
const factory = createGaussianContentAdapterFactory({
  createSortBackend: () => ({
    sort: async () => new Uint32Array(),
    dispose() {},
  }),
});
const adapter = factory({
  renderer: left,
  submissions: createSubmissionScheduler({ scheduleRender() {} }),
  scheduleRender() {},
});
const gaussian = left.getActors()[0];
const gaussianMapper = gaussian.getMapper();
const gaussianData = gaussianMapper.getInputData();
gaussianData.getPoints().setData(Float32Array.of(-0.5, 0, 0), 3);
gaussianData
  .getPointData()
  .getArrayByName("GaussianScale")
  .setData(Float32Array.of(0.15, 0.15, 0.15), 3);
gaussianData
  .getPointData()
  .getArrayByName("GaussianRotation")
  .setData(Float32Array.of(1, 0, 0, 0), 4);
gaussianData
  .getPointData()
  .getArrayByName("GaussianColor")
  .setData(Float32Array.of(1, 0, 0, 0.5), 4);
gaussianMapper.getPointOrder().setData(Uint32Array.of(0), 1);
gaussian.setVisibility(true);

const plane = vtkPlaneSource.newInstance({
  origin: [0.25, -0.25, 0],
  point1: [0.75, -0.25, 0],
  point2: [0.25, 0.25, 0],
});
const meshMapper = vtkMapper.newInstance();
meshMapper.setInputConnection(plane.getOutputPort());
const mesh = vtkActor.newInstance();
mesh.setMapper(meshMapper);
mesh.getProperty().setLighting(false);
mesh.getProperty().setColor(0, 0, 1);
mesh.getProperty().setOpacity(0.5);
left.addActor(mesh);

// The same mapper class can draw ordinary straight-alpha simple points.
const points = vtkPolyData.newInstance();
points.getPoints().setData(Float32Array.of(0, 0, 0), 3);
const pointsMapper = vtkPointGaussianMapper.newInstance({
  scaleFactor: 0,
  emissive: false,
});
pointsMapper.setInputData(points);
const pointActor = vtkActor.newInstance();
pointActor.setMapper(pointsMapper);
pointActor.getProperty().setPointSize(16);
pointActor.getProperty().setColor(0, 1, 0);
pointActor.getProperty().setOpacity(0.5);
right.addActor(pointActor);

window.fixture = {
  capture() {
    renderWindow.render();
    const pixel = (x, y) => {
      const rgba = new Uint8Array(4);
      gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      return Array.from(rgba);
    };
    return {
      gaussian: pixel(20, 40),
      mesh: pixel(60, 40),
      point: pixel(120, 40),
    };
  },
  dispose() {
    adapter.dispose();
    factory.dispose(left);
    generic.delete();
  },
};
renderWindow.render();
document.documentElement.dataset.ready = "true";
