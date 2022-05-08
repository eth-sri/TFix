module.exports = {
  target: "node",
  resolve: {
    // to help webpack work in bazel's sandbox
    symlinks: false,
  },
  mode: "none",
};
