const YAML = require('yaml');
const JSON5 = require('json5');

const INVALID_CONFIG = 'INVALID';
const VALID_CONFIG = 'SUCCESS';
const NO_CONFIG = 'NONE';

const defaultConfig = {
  'parserOptions': {
    'ecmaVersion': 8,
    'ecmaFeatures': {'jsx': true, 'experimentalObjectRestSpread': true},
    'sourceType': 'module'
  },
  // enable all global variables
  // http://eslint.org/docs/user-guide/configuring#specifying-environments
  'envs': {
    'browser': true,
    'node': true,
    'commonjs': true,
    //   "es6": true,
    'worker': true,
    'amd': true,
    'mocha': true,
    'jasmine': true,
    'jest': true,
    'phantomjs': true,
    'protractor': true,
    'qunit': true,
    'jquery': true,
    'prototypejs': true,
    'shelljs': true,
    'meteor': true,
    'mongo': true,
    'applescript': true,
    'nashorn': true,
    'serviceworker': true,
    'atomtest': true,
    'embertest': true,
    'webextensions': true,
    'greasemonkey': true
  },
  'extends': 'google',
  'rules': {
    'linebreak-style': 'off',
    'no-tabs': 'off',
    'object-curly-spacing': 'off',
    'space-in-parens': 'off',
    'spaced-comment': 'off',
    'space-infix-ops': 'off',
    'spaced-comment': 'off',
    'max-len': 'off',
    'array-bracket-spacing': 'off',
    'padded-blocks': 'off',
    'no-multi-spaces': 'off',
    'func-call-spacing': 'off',
    'space-before-function-paren': 'off',
    'quote-props': 'off',
    'no-multiple-empty-lines': 'off',
    'comma-spacing': 'off',
    'eol-last': 'off',
    'switch-colon-spacing': 'off',
    'no-trailing-spaces': 'off',
    'block-spacing': 'off',
    'brace-style': 'off',
    'quotes': 'off',
    'comma-dangle': 'off',
    'space-before-blocks': 'off',
    'key-spacing': 'off',
    'camelcase': 'off',
    'new-cap': 'off',
    'keyword-spacing': 'off',
    'semi-spacing': 'off',
    'semi': 'never',
    'computed-property-spacing': 'never',
    'one-var': 'never',
    'no-var': 'never',
    'require-jsdoc': 'off',
    'no-unused-vars':
        'off',  // seems to spammy for things like react and in general for function etc.
    'curly': 'multi-line',
    'arrow-parens': 'off'
  }
};




