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
  "SEPTIC_ARTHRITIS_MODULE",
  "BACTERIAL_MENINGITIS_MODULE",
  "ENCEPHALITIS_MODULE",
  "SPINAL_EPIDURAL_ABSCESS_MODULE",
  "BRAIN_ABSCESS_MODULE",
  "NECROTIZING_SOFT_TISSUE_INFECTION_MODULE",
  "PJI_MODULE",
];

const ENRICHED_SOURCE_URLS = {
  "Margaretten et al. JAMA|2007": "https://pubmed.ncbi.nlm.nih.gov/17405973/",
  "Mathews et al. Lancet|2010": "https://pubmed.ncbi.nlm.nih.gov/20206778/",
  "Carpenter et al. Acad Emerg Med|2011": "https://pubmed.ncbi.nlm.nih.gov/21843213/",
  "Shirtliff and Mader. Am Fam Physician|2002": "https://www.aafp.org/pubs/afp/issues/2002/0401/p1341.html",
  "Attia et al. JAMA|1999": "https://pubmed.ncbi.nlm.nih.gov/10411200/",
  "van de Beek et al. N Engl J Med|2004": "https://pubmed.ncbi.nlm.nih.gov/15509818/",
  "Brouwer et al. Lancet|2021": "https://pubmed.ncbi.nlm.nih.gov/34303412/",
  "Viallon et al. Clin Infect Dis|1999": "https://pubmed.ncbi.nlm.nih.gov/10451174/",
  "Spanos et al. JAMA|1989": "https://pubmed.ncbi.nlm.nih.gov/2810603/",
  "Sakushima et al. J Infect|2011": "https://pubmed.ncbi.nlm.nih.gov/21194480/",
  "Tunkel et al. IDSA|2008": "https://www.idsociety.org/practice-guideline/encephalitis/",
  "Binnicker et al. J Clin Virol|2022": "https://pubmed.ncbi.nlm.nih.gov/37390620/",
  "Kneen et al. Clin Radiol|2016": "https://pubmed.ncbi.nlm.nih.gov/27185323/",
  "Arko et al. Surg Neurol Int|2014": "https://pubmed.ncbi.nlm.nih.gov/25081964/",
  "Davis et al. J Emerg Med|2011": "https://pubmed.ncbi.nlm.nih.gov/15028325/",
  "Tetsuka et al. Neurol Med Chir|2019": "https://pubmed.ncbi.nlm.nih.gov/33324773/",
  "Brouwer et al. Curr Opin Infect Dis|2017": "https://pubmed.ncbi.nlm.nih.gov/27828809/",
  "Helweg-Larsen et al. Open Forum Infect Dis|2018": "https://pubmed.ncbi.nlm.nih.gov/29909695/",
  "StatPearls Brain Abscess|2024": "https://www.ncbi.nlm.nih.gov/books/NBK441841/",
  "Kim et al. AJNR|2002": "https://pubmed.ncbi.nlm.nih.gov/9843275/",
  "IDSA SSTI Guideline|2014": "https://www.idsociety.org/practice-guideline/skin-and-soft-tissue-infections/",
  "Fernando et al. Ann Surg|2019": "https://pubmed.ncbi.nlm.nih.gov/29672405/",
  "MRI and CT Systematic Review|2021": "https://pubmed.ncbi.nlm.nih.gov/34302500/",
};

function attachSourceUrls(modules) {
  const enrich = (source) => {
    if (!source || source.url) return source;
    const key = `${source.short}|${source.year ?? ""}`;
    const url = ENRICHED_SOURCE_URLS[key];
    return url ? { ...source, url } : source;
  };

  return modules.map((module) => ({
    ...module,
    pretestPresets: (module.pretestPresets || []).map((preset) => ({
      ...preset,
      source: enrich(preset.source),
    })),
    items: (module.items || []).map((item) => ({
      ...item,
      source: enrich(item.source),
    })),
  }));
}

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

  const modules = attachSourceUrls(sandbox.module.exports);
  if (!Array.isArray(modules)) {
    throw new Error("Extraction failed: exports is not an array");
  }

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(modules, null, 2));
  console.log(`Wrote ${modules.length} modules to ${OUT}`);
}

main();
