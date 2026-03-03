const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SRC = path.resolve(process.cwd(), "ProbID/lrSyndromes.ts");
const OUT = path.resolve(process.cwd(), "backend/app/data/probid_modules.json");

const MODULE_EXPORTS = [
  "CAP_MODULE",
  "VAP_MODULE",
  "CDI_MODULE",
  "UTI_MODULE",
  "ENDO_MODULE",
  "ACTIVE_TB_MODULE",
  "PJP_MODULE",
  "INVASIVE_CANDIDIASIS_MODULE",
  "INVASIVE_MOLD_MODULE",
  "PJI_MODULE",
];

function findModuleObject(source, exportName) {
  const marker = `export const ${exportName}: SyndromeLRModule =`;
  const start = source.indexOf(marker);
  if (start === -1) {
    throw new Error(`Could not find export marker for ${exportName}`);
  }

  const eqIdx = source.indexOf("=", start);
  const braceStart = source.indexOf("{", eqIdx);
  if (braceStart === -1) {
    throw new Error(`Could not find object start for ${exportName}`);
  }

  let i = braceStart;
  let depth = 0;
  let inSingle = false;
  let inDouble = false;
  let inTemplate = false;
  let inLineComment = false;
  let inBlockComment = false;
  let escaped = false;

  for (; i < source.length; i++) {
    const ch = source[i];
    const next = source[i + 1];

    if (inLineComment) {
      if (ch === "\n") inLineComment = false;
      continue;
    }
    if (inBlockComment) {
      if (ch === "*" && next === "/") {
        inBlockComment = false;
        i++;
      }
      continue;
    }

    if (inSingle) {
      if (!escaped && ch === "'") inSingle = false;
      escaped = !escaped && ch === "\\";
      continue;
    }
    if (inDouble) {
      if (!escaped && ch === '"') inDouble = false;
      escaped = !escaped && ch === "\\";
      continue;
    }
    if (inTemplate) {
      if (!escaped && ch === "`") inTemplate = false;
      escaped = !escaped && ch === "\\";
      continue;
    }

    if (ch === "/" && next === "/") {
      inLineComment = true;
      i++;
      continue;
    }
    if (ch === "/" && next === "*") {
      inBlockComment = true;
      i++;
      continue;
    }
    if (ch === "'") {
      inSingle = true;
      escaped = false;
      continue;
    }
    if (ch === '"') {
      inDouble = true;
      escaped = false;
      continue;
    }
    if (ch === "`") {
      inTemplate = true;
      escaped = false;
      continue;
    }

    if (ch === "{") {
      depth++;
    } else if (ch === "}") {
      depth--;
      if (depth === 0) {
        const objectLiteral = source.slice(braceStart, i + 1);
        return objectLiteral;
      }
    }
  }

  throw new Error(`Failed to parse balanced object literal for ${exportName}`);
}

function buildEvalProgram(source) {
  const blocks = MODULE_EXPORTS.map((name) => `const ${name} = ${findModuleObject(source, name)};`);
  const exportLine = `module.exports = [${MODULE_EXPORTS.join(", ")}];`;
  return `${blocks.join("\n\n")}\n\n${exportLine}\n`;
}

function main() {
  const source = fs.readFileSync(SRC, "utf8");
  const program = buildEvalProgram(source);
  const sandbox = { module: { exports: null } };
  vm.createContext(sandbox);
  vm.runInContext(program, sandbox, { timeout: 2000 });

  const modules = sandbox.module.exports;
  if (!Array.isArray(modules)) {
    throw new Error("Extraction failed: exports is not an array");
  }

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(modules, null, 2));
  console.log(`Wrote ${modules.length} modules to ${OUT}`);
}

main();

