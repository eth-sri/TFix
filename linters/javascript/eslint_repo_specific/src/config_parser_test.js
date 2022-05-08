/*
 *  Copyright 2019 DeepCode AG
 *
 *  Author: Jan Eberhardt
 */

const assert = require("assert");
const index = require("./config_parser.js");

const jsonConfigString =
  "{\n" +
  '    "env": {\n' +
  '        "browser": true\n' +
  "    },\n" +
  '    "extends": "eslint:recommended",\n' +
  '    "parserOptions": {\n' +
  '        "ecmaVersion": 5\n' +
  "    },\n" +
  '    "rules": {\n' +
  '        "indent": [\n' +
  '            "error",\n' +
  '            "tab"\n' +
  "        ],\n" +
  '        "linebreak-style": [\n' +
  '            "error",\n' +
  '            "unix"\n' +
  "        ],\n" +
  '        "quotes": [\n' +
  '            "error",\n' +
  '            "double"\n' +
  "        ],\n" +
  '        "semi": [\n' +
  '            "error",\n' +
  '            "never"\n' +
  "        ]\n" +
  "    }\n" +
  "}\n";

// this exports something that is not valid json but a javascript object
const jsConfigString =
  "module.exports = {\n" +
  "    env: {\n" +
  "        browser: true\n" +
  "    },\n" +
  '    extends: "eslint:recommended",\n' +
  "    parserOptions: {\n" +
  "        ecmaVersion: 5,\n" +
  "    },\n" +
  "    rules: {\n" +
  "        indent: [\n" +
  '            "error",\n' +
  '            "tab",\n' +
  "        ],\n" +
  '        "linebreak-style": [\n' +
  '            "error",\n' +
  '            "unix",\n' +
  "        ],\n" +
  "        quotes: [\n" +
  '            "error",\n' +
  '            "double",\n' +
  "        ],\n" +
  "        semi: [\n" +
  '            "error",\n' +
  '            "never",\n' +
  "        ],\n" +
  "    }\n" +
  "}\n";

// this is from a real repository
const jsConfigString2 =
  "module.exports = {\n" +
  "  extends: ['standard', 'prettier'],\n" +
  "  plugins: ['prettier'],\n" +
  "  rules: {\n" +
  "    'prettier/prettier': [\n" +
  "      'error',\n" +
  "      {\n" +
  "        singleQuote: true,\n" +
  "        semi: false,\n" +
  "      },\n" +
  "    ]\n" +
  "  },\n" +
  "  env: { mocha: true }\n" +
  "};";

// this is the same as jsonConfigString2, just in json format
const jsonConfigString2 =
  "{\n" +
  '  "extends": ["standard", "prettier"],\n' +
  '  "plugins": ["prettier"],\n' +
  '  "rules": {\n' +
  '    "prettier/prettier": [\n' +
  '      "error",\n' +
  "      {\n" +
  '        "singleQuote": true,\n' +
  '        "semi": false\n' +
  "      }\n" +
  "    ]\n" +
  "  },\n" +
  '  "env": {\n' +
  '    "mocha": true\n' +
  "  }\n" +
  "}";

const ymlConfigString =
  "env:\n" +
  "  browser: true\n" +
  "extends: 'eslint:recommended'\n" +
  "parserOptions:\n" +
  "  ecmaVersion: 5\n" +
  "rules:\n" +
  "  indent:\n" +
  "    - error\n" +
  "    - tab\n" +
  "  linebreak-style:\n" +
  "    - error\n" +
  "    - unix\n" +
  "  quotes:\n" +
  "    - error\n" +
  "    - double\n" +
  "  semi:\n" +
  "    - error\n" +
  "    - never\n";

const config = JSON.parse(jsonConfigString);

function testParseInvalidConfig() {
  assert.deepEqual(
    index.defaultConfig,
    index.parseConfig("invalid config").config
  );
  assert.equal("INVALID", index.parseConfig("invalid config").configStatus);
}

function testParseDefaultConfig() {
  assert.deepEqual(index.defaultConfig, index.parseConfig("").config);
  assert.equal("NONE", index.parseConfig("").configStatus);
}

function testParseJsonConfig() {
  const toBeParsed = JSON.stringify({
    format: ".json",
    content: jsonConfigString,
  });
  assert.deepEqual(config, index.parseConfig(toBeParsed).config);
  assert.equal("SUCCESS", index.parseConfig(toBeParsed).configStatus);
}

function testParseJsConfig() {
  const toBeParsed = JSON.stringify({
    format: ".js",
    content: "module.exports = " + jsonConfigString,
  });
  assert.deepEqual(config, index.parseConfig(toBeParsed).config);
  assert.equal("SUCCESS", index.parseConfig(toBeParsed).configStatus);
}

function testParseJsConfig2() {
  const toBeParsed = JSON.stringify({ format: ".js", content: jsConfigString });
  assert.deepEqual(config, index.parseConfig(toBeParsed).config);
  assert.equal("SUCCESS", index.parseConfig(toBeParsed).configStatus);
}

function testParseJsConfig3() {
  const jsonToBeParsed = JSON.stringify({
    format: ".json",
    content: jsonConfigString2,
  });
  const jsToBeParsed = JSON.stringify({
    format: ".js",
    content: jsConfigString2,
  });
  assert.deepEqual(
    index.parseConfig(jsonToBeParsed).config,
    index.parseConfig(jsToBeParsed).config
  );
  assert.equal("SUCCESS", index.parseConfig(jsonToBeParsed).configStatus);
  assert.equal("SUCCESS", index.parseConfig(jsToBeParsed).configStatus);
}

function testParseYmlConfig() {
  const toBeParsed = JSON.stringify({
    format: ".yml",
    content: ymlConfigString,
  });
  assert.deepEqual(config, index.parseConfig(toBeParsed).config);
  assert.equal("SUCCESS", index.parseConfig(toBeParsed).configStatus);
}

// run all the tests
testParseInvalidConfig();
testParseDefaultConfig();
testParseJsonConfig();
testParseJsConfig();
testParseJsConfig2();
testParseJsConfig3();
testParseYmlConfig();
