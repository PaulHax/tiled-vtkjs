import { makeMapper } from "./vtkStub";

export default {
  newInstance: (options: { scaleFactor: number }) => {
    if (options.scaleFactor !== 0)
      throw new Error("Point tiles require scaleFactor=0");
    return makeMapper();
  },
};
