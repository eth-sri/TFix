/*
 *  Copyright 2019 DeepCode AG
 *
 *  Author: Jan Eberhardt
 */

const YAML = require("yaml");
const JSON5 = require("json5");

const INVALID_CONFIG = "INVALID";
const VALID_CONFIG = "SUCCESS";
const NO_CONFIG = "NONE";

const defaultConfig = {
  parserOptions: {
    ecmaVersion: 8,
    ecmaFeatures: { jsx: true, experimentalObjectRestSpread: true },
    sourceType: "module",
  },
  // enable all global variables
  // http://eslint.org/docs/user-guide/configuring#specifying-environments
  envs: {
    browser: true,
    node: true,
    commonjs: true,
    //   "es6": true,
    worker: true,
    amd: true,
    mocha: true,
    jasmine: true,
    jest: true,
    phantomjs: true,
    protractor: true,
    qunit: true,
    jquery: true,
    prototypejs: true,
    shelljs: true,
    meteor: true,
    mongo: true,
    applescript: true,
    nashorn: true,
    serviceworker: true,
    atomtest: true,
    embertest: true,
    webextensions: true,
    greasemonkey: true,
  },
  extends: "google",
  rules: {
    "linebreak-style": "off",
    "no-tabs": "off",
    "object-curly-spacing": "off",
    "space-in-parens": "off",
    "spaced-comment": "off",
    "space-infix-ops": "off",
    "spaced-comment": "off",
    "max-len": "off",
    "array-bracket-spacing": "off",
    "padded-blocks": "off",
    "no-multi-spaces": "off",
    "func-call-spacing": "off",
    "space-before-function-paren": "off",
    "quote-props": "off",
    "no-multiple-empty-lines": "off",
    "comma-spacing": "off",
    "eol-last": "off",
    "switch-colon-spacing": "off",
    "no-trailing-spaces": "off",
    "block-spacing": "off",
    "brace-style": "off",
    quotes: "off",
    "comma-dangle": "off",
    "space-before-blocks": "off",
    "key-spacing": "off",
    camelcase: "off",
    "new-cap": "off",
    "keyword-spacing": "off",
    "semi-spacing": "off",
    semi: "never",
    "computed-property-spacing": "never",
    "one-var": "never",
    "no-var": "never",
    "require-jsdoc": "off",
    "no-unused-vars": "off", // seems to spammy for things like react and in general for function etc.
    curly: "multi-line",
    "arrow-parens": "off",
  },
};

function parseJsonConfig(configContent) {
  try {
    return {
      config: JSON.parse(configContent),
      configStatus: VALID_CONFIG,
      defaultConfig: defaultConfig,
    };
  } catch (e) {
    // if the parsing of the config failed, just use the default config
    return {
      config: defaultConfig,
      configStatus: INVALID_CONFIG,
      defaultConfig: defaultConfig,
    };
  }
}

function parseJsConfig(configContent) {
  // This might seem a bit hacky but we do not want to execute any javascript
  // code from some random repository. The javascript file only exports a json
  // object. The current approach is to take the substring that represents the
  // json object and interpret it as a json config.
  const begin = configContent.indexOf("{");
  const end = configContent.lastIndexOf("}");
  if (begin !== -1 && end !== -1 && end < configContent.length) {
    let configJsonString = configContent.substring(begin, end + 1);
    try {
      return {
        config: JSON5.parse(configJsonString),
        configStatus: VALID_CONFIG,
        defaultConfig: defaultConfig,
      };
    } catch (e) {
      // if the parsing of the config failed, just use the default config
      return {
        config: defaultConfig,
        configStatus: INVALID_CONFIG,
        defaultConfig: defaultConfig,
      };
    }
  } else {
    return {
      config: defaultConfig,
      configStatus: INVALID_CONFIG,
      defaultConfig: defaultConfig,
    };
  }
}

function parseYmlConfig(configContentObj) {
  try {
    return {
      config: YAML.parse(configContentObj),
      configStatus: VALID_CONFIG,
      defaultConfig: defaultConfig,
    };
  } catch (e) {
    return {
      config: defaultConfig,
      configStatus: INVALID_CONFIG,
      defaultConfig: defaultConfig,
    };
  }
}

function parseConfig(configContent) {
  if (configContent === "") {
    return {
      config: defaultConfig,
      configStatus: NO_CONFIG,
      defaultConfig: defaultConfig,
    };
  }
  let configContentObj;
  const parseFailRes = {
    config: defaultConfig,
    configStatus: INVALID_CONFIG,
    defaultConfig: defaultConfig,
  };
  try {
    configContentObj = JSON.parse(configContent);
  } catch (e) {
    return parseFailRes;
  }
  if (
    !configContentObj.hasOwnProperty("format") ||
    !configContentObj.hasOwnProperty("content")
  ) {
    return parseFailRes;
  }
  let config = parseFailRes;
  if (configContentObj.format === ".json") {
    config = parseJsonConfig(configContentObj.content);
  } else if (configContentObj.format === ".js") {
    config = parseJsConfig(configContentObj.content);
  } else if (configContentObj.format === ".yml") {
    config = parseYmlConfig(configContentObj.content);
  }
  return config;
}

module.exports.parseConfig = parseConfig;

// visible for testing
module.exports.defaultConfig = defaultConfig;
