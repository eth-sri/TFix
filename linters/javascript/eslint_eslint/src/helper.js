const crypto = require("crypto");

function ParseException(message) {
  this.message = message;
  this.name = "ParseException";
}

function GetError(messages) {
  // If the operation ends with a parsing error, you will get a single message
  // for this file, with fatal: true added as an extra property.
  if (messages.length == 1 && messages[0].hasOwnProperty("fatal")) {
    let msg = messages[0].message;
    if (msg.startsWith("Parsing error: ")) {
      // Remove prefix as we add similar prefix in serving.
      msg = msg.substr("Parsing error: ".length);
    }
    return msg;
  }
  return null;
}

function Trim(text) {
  if (text.length < 100) {
    return text;
  }
  return text.substr(0, 100) + "...";
}

const messages = new Map([
  ["require-jsdoc", "Missing JSDoc comment."],
  ["valid-jsdoc", "Missing JSDoc for parameter or return value."],
  ["no-unused-vars", "Unused variable."],
  ["no-undef", "Undefined variable."],
  ["no-useless-escape", "Unnecessary escape character."],
  ["no-redeclare", "Redeclaring variable."],
  [
    "no-control-regex",
    "Unexpected control character(s) in regular expression.",
  ],
]);

function MessageToReport(message) {
  // create the rule id from the message_id and the message
  // otherwise this leads to weird grouping later on.
  // This is to distinguish reports like: "'process' is not defined." and
  // "'__dirname' is not defined." (they have the same eslint message id)
  let msg = message["message"];
  if (messages.has(message["ruleId"])) {
    msg = messages.get(message["ruleId"]);
  }
  // const message_hash = crypto.createHash("md5").update(msg).digest("hex");
  const ruleId = message["ruleId"];

  var result = {
    rule_id: ruleId,
    message: msg,

    // "evidence":     Trim(message["source"]),

    line_begin: message["line"],
    col_begin: message["column"],

    //        "severity":     message["severity"] ,
    // severity reported by eslint is not useful so we just hard code a value.
    severity: 0.5,
  };

  if (msg !== message["message"]) {
    result.position_message = message["message"];
  }

  if (
    message.hasOwnProperty("endLine") &&
    message.hasOwnProperty("endColumn") &&
    message.endLine == message.line
  ) {
    result.col_end = message.endColumn - 1;
  } else {
    result.col_begin = result.col_begin - (result.col_begin <= 1 ? 0 : 1);
    result.col_end = 1000;
  }
  result.line_end = message.line;

  return result;
}

exports.run = function (argv, cli) {
  const runner = require("./base");

  // config will be ignored here...
  function processFile(fileName, file_content, config) {
    var report = cli.executeOnText(file_content);

    let messages = report["results"][0]["messages"];
    var error = GetError(messages);
    if (error) {
      throw new ParseException(error);
    }
    var results = [];
    for (var i = 0; i < messages.length; i++) {
      results.push(MessageToReport(messages[i]));
    }

    return [results];
  }

  function printUsage() {
    console.log("Usage: node parse.js <file> or node parse.js - JSON.");
  }

  runner.run(argv, processFile, printUsage);
};

exports.runWithCb = function (argv, getEngineAndConfigs) {
  const runner = require("./base");

  function processFile(file_name, file_content, config_content) {
    let report;
    let engineCb, configs;
    try {
      [engineCb, configs] = getEngineAndConfigs(config_content);
      report = engineCb(configs.config).executeOnText(file_content);
    } catch (err) {
      // fall back to default config in case something went wrong
      if (
        configs.configStatus === "INVALID" ||
        configs.configStatus === "NONE"
      ) {
        // we are running on the default config, no need to rerun, the error was
        // not related to the config.
        throw err;
      }
      configs.configStatus = "INVALID";
      report = engineCb(configs.defaultConfig).executeOnText(file_content);
    }

    let messages = report["results"][0]["messages"];
    var error = GetError(messages);
    if (error) {
      throw new ParseException(error);
    }
    var results = [];
    for (var i = 0; i < messages.length; i++) {
      results.push(MessageToReport(messages[i]));
    }

    return [results, configs ? configs.configStatus : "INVALID"];
  }

  function printUsage() {
    console.log("Usage: node parse.js <file> or node parse.js - JSON.");
  }

  runner.run(argv, processFile, printUsage);
};
