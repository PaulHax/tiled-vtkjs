import { makeMapper } from "./vtkStub";

export default {
  newInstance: () => ({
    ...makeMapper(),
    setSplatShaderCode(_code: string) {},
    setPointOrder(_order: unknown) {},
  }),
};
