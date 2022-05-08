/*
 *  Copyright 2019 DeepCode AG
 *
 *  Author: Jan Eberhardt
 */

const CLIEngine = require("eslint").CLIEngine;
// documentation: http://eslint.org/docs/developer-guide/nodejs-api

const configParser = require("./config_parser");

function getEngineWithConfig(parsedConfig) {
  return new CLIEngine({ baseConfig: parsedConfig });
}

function getEngineAndConfigs(configContent) {
  const parsedConfig = configParser.parseConfig(configContent);
  return [getEngineWithConfig, parsedConfig];
}

const runner = require("./helper");
runner.runWithCb(process.argv, getEngineAndConfigs);
