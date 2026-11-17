export default {
  extend: (pass: object) => {
    let operation: string | null = null;
    Object.assign(pass, {
      setCurrentOperation(value: string) {
        operation = value;
      },
      getOperation: () => operation,
      delete() {},
    });
  },
};
