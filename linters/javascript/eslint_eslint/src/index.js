var CLIEngine = require("eslint").CLIEngine;
// documentation: http://eslint.org/docs/developer-guide/nodejs-api

const options = {
  plugins: ["node"],
  extends: ["eslint:recommended", "plugin:node/recommended"],
  parserOptions: {
    sourceType: "module",
    ecmaFeatures: { jsx: true, experimentalObjectRestSpread: true },
  },
  rules: { "linebreak-style": "off", "no-tabs": "off" },
};

var cli = new CLIEngine({ baseConfig: options });

const runner = require("./helper");
runner.run(process.argv, cli);
