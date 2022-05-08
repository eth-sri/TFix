function isMinified(file_content) {
  var content = file_content.substr(0, 500);
  var content_length = content.length;

  var lines_match = content.match(/\n/g);
  if (lines_match == null) return true;
  var num_lines = lines_match.length;

  var spaces_match = content.match(/\s/g);
  if (spaces_match == null) return true;
  var num_spaces = spaces_match.length;

  var spaces_ratio = content_length / num_spaces;
  var lines_ratio = content_length / num_lines;

  return lines_ratio > 300 || spaces_ratio > 50;
}

exports.run = function (argv, file_cb, help_cb) {
  // Normalize arguments by removing '--' arg, if present.
  if (argv.length >= 3 && argv[2] === "--") {
    argv.splice(2, 1);
  }

  // first two args are always node binary and executed javascript file.
  if (argv.length === 3) {
    const program = argv[2];
    try {
      const fs = require("fs");
      const file_content = fs.readFileSync(program, { encoding: "utf8" });
      const result = file_cb(program, file_content, "");

      console.log(result);
      // console.log(JSON.stringify(result));
    } catch (err) {
      console.error(err);
      process.exit(-1);
    }
  } else if (argv.length === 4 && argv[2] === "-" && argv[3] === "JSON") {
    const readline = require("readline");
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false,
    });
    let errorOccured = false;
    rl.on("line", (line) => {
      if (line.length < 5 && line.trim().length == 0) {
        // Empty lines.
        console.error("Skipping empty line: " + line);
        return;
      }
      const arrayOfProgs = JSON.parse(line);
      arrayOfProgs.forEach((prog) => {
        if (prog.hasOwnProperty("force_fail")) {
          console.error("Forced failure! Used in testing.");
          process.exit(-1);
        }
        let config = "";
        if (prog.hasOwnProperty("config")) {
          config = prog.config;
          delete prog.config;
        }
        if (prog.hasOwnProperty("code")) {
          const code = prog.code;

          if (isMinified(code)) {
            prog.data = "minified";
            delete prog.code;
            console.log(JSON.stringify(prog));
          } else {
            try {
              const [result, config_status] = file_cb(
                prog.file_name,
                code,
                config
              );
              prog.data = result;
              delete prog.code;
              if (config_status) {
                prog.config_status = config_status;
              }
              console.log(JSON.stringify(prog));
            } catch (err) {
              errorOccured = true;
              if (err.message) {
                prog.parse_error = err.message;
              }
              prog.data = "error";
              delete prog.code;
              console.log(JSON.stringify(prog));
            }
          }
        } else {
          console.error("Expected 'code' attribute but none found in: " + line);
        }
      });
    });

    process.on("exit", function () {
      process.exit(errorOccured ? -1 : 0);
    });
  } else {
    help_cb();
    process.exit();
  }
};
