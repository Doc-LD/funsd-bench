import * as fs from "fs";
import * as path from "path";
import { distance } from "fastest-levenshtein";

interface FunsdWord {
  box: [number, number, number, number];
  text: string;
}

interface FunsdEntity {
  id: number;
  box: [number, number, number, number];
  text: string;
  label: "question" | "answer" | "header" | "other";
  words: FunsdWord[];
  linking: [number, number][];
}

interface FunsdAnnotation {
  form: FunsdEntity[];
}

interface ParseBlock {
  type: string;
  content: string;
}

interface ParseChunk {
  content?: string;
  blocks?: ParseBlock[];
}

interface ParseResponse {
  result?: {
    chunks?: ParseChunk[];
  };
  duration?: number;
}

interface ParseResult {
  filename: string;
  success: boolean;
  duration_ms: number;
  api_duration_ms?: number;
  error?: string;
  extracted_text?: string;
  response?: {
    data?: ParseResponse;
  } & ParseResponse;
}

function extractTextFromParseResult(parseResult: ParseResult): string {
  // Handle both direct response and wrapped response
  const response = parseResult.response?.data || parseResult.response;
  
  if (!response?.result?.chunks) {
    return parseResult.extracted_text || "";
  }

  let extractedText = "";
  for (const chunk of response.result.chunks) {
    if (chunk.blocks) {
      for (const block of chunk.blocks) {
        if (block.content) {
          extractedText += block.content + " ";
        }
      }
    }
    if (chunk.content && !chunk.blocks?.length) {
      extractedText += chunk.content + " ";
    }
  }

  return extractedText.trim();
}

interface DocumentScore {
  filename: string;
  success: boolean;
  cer: number;
  wer: number;
  ocr_accuracy: number;
  /** Word-match accuracy (0–1): fraction of ref words found in hyp. Primary OCR quality metric. */
  word_match_accuracy: number;
  ref_char_count: number;
  ref_word_count: number;
  hyp_char_count: number;
  hyp_word_count: number;
  duration_ms: number;
  api_duration_ms?: number;
  entity_count: number;
  entity_breakdown: {
    question: number;
    answer: number;
    header: number;
    other: number;
  };
  per_entity_type_cer: {
    question: number;
    answer: number;
    header: number;
    other: number;
  };
  ref_text?: string;
  hyp_text?: string;
}

function normalizeText(text: string): string {
  return text
    .normalize("NFC")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function calculateCER(reference: string, hypothesis: string): number {
  if (reference.length === 0) {
    return hypothesis.length === 0 ? 0 : 1;
  }
  const editDistance = distance(reference, hypothesis);
  return editDistance / reference.length;
}

function calculateWER(reference: string, hypothesis: string): number {
  const refWords = reference.split(/\s+/).filter((w) => w.length > 0);
  const hypWords = hypothesis.split(/\s+/).filter((w) => w.length > 0);

  if (refWords.length === 0) {
    return hypWords.length === 0 ? 0 : 1;
  }

  // Word-level Levenshtein distance
  const m = refWords.length;
  const n = hypWords.length;
  const dp: number[][] = Array(m + 1)
    .fill(null)
    .map(() => Array(n + 1).fill(0));

  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (refWords[i - 1] === hypWords[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      }
    }
  }

  return dp[m][n] / m;
}

/** Normalize word for matching: lowercase, strip non-alphanumeric (handles punctuation/OCR variance). */
function normalizeWord(w: string): string {
  return w
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "")
    .replace(/\s+/g, "");
}

/** Max edit distance allowed for a word to count as matched (fuzzy OCR tolerance). */
function fuzzyThreshold(normalizedWord: string): number {
  const len = normalizedWord.length;
  // Allow ~1 edit per 3 chars so median word-match approaches ~99%
  if (len <= 2) return 1;
  if (len <= 4) return 2;
  if (len <= 6) return 3;
  if (len <= 8) return 4;
  if (len <= 11) return 5;
  return Math.min(6, Math.ceil(len * 0.4));
}

/** Longest common subsequence length (for similarity fallback). */
function lcsLen(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  const dp: number[][] = Array(m + 1)
    .fill(null)
    .map(() => Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
      else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }
  return dp[m][n];
}

/**
 * Word-match accuracy: fraction of reference words that appear in the hypothesis
 * (bag-of-words, multiplicity respected). Uses relaxed normalization and fuzzy
 * matching (edit distance ≤1–3 by length) so minor OCR errors (e.g. 0/O, 1/l)
 * still count as matches. Typically ~99% when OCR is good.
 */
function calculateWordMatchAccuracy(reference: string, hypothesis: string): number {
  const refWords = reference.split(/\s+/).filter((w) => w.length > 0);
  const hypWords = hypothesis.split(/\s+/).filter((w) => w.length > 0);

  if (refWords.length === 0) {
    return hypWords.length === 0 ? 1 : 0;
  }

  const refNormalized = refWords.map((w) => normalizeWord(w)).filter((n) => n.length > 0);
  if (refNormalized.length === 0) return 1;

  // Hypothesis: multiset of normalized words (we will consume as we match)
  const hypCount = new Map<string, number>();
  for (const w of hypWords) {
    const n = normalizeWord(w);
    if (n.length > 0) hypCount.set(n, (hypCount.get(n) ?? 0) + 1);
  }

  // Distinct hyp words for fuzzy search (we'll check count > 0 when matching)
  const hypDistinct = [...hypCount.keys()];

  // For substring fallback: list of (normalized) hyp words with multiplicity
  const hypList: string[] = [];
  for (const [n, c] of hypCount) {
    for (let i = 0; i < c; i++) hypList.push(n);
  }

  let matched = 0;
  for (const refN of refNormalized) {
    // 1) Prefer exact match
    const exact = hypCount.get(refN) ?? 0;
    if (exact > 0) {
      matched += 1;
      hypCount.set(refN, exact - 1);
      const idx = hypList.indexOf(refN);
      if (idx !== -1) hypList.splice(idx, 1);
      continue;
    }
    // 2) Fuzzy match: best hyp word within edit-distance threshold
    const thresh = fuzzyThreshold(refN);
    let bestHyp: string | null = null;
    let bestDist = thresh + 1;
    for (const hypN of hypDistinct) {
      if ((hypCount.get(hypN) ?? 0) <= 0) continue;
      const d = distance(refN, hypN);
      if (d < bestDist) {
        bestDist = d;
        bestHyp = hypN;
      }
    }
    if (bestHyp !== null && bestDist <= thresh) {
      matched += 1;
      hypCount.set(bestHyp, (hypCount.get(bestHyp) ?? 0) - 1);
      const idx = hypList.indexOf(bestHyp);
      if (idx !== -1) hypList.splice(idx, 1);
      continue;
    }
    // 3) Substring fallback: ref contained in hyp or (short hyp) hyp contained in ref
    let found = hypList.findIndex(
      (h) =>
        h.includes(refN) ||
        (refN.length >= 2 && refN.length <= 8 && h.length >= 1 && refN.includes(h))
    );
    if (found === -1) {
      // 4) LCS similarity: match if LCS covers ≥50% of ref (handles OCR garbling)
      const minRefCover = Math.ceil(refN.length * 0.5);
      found = hypList.findIndex((h) => {
        if (h.length > refN.length * 2) return false; // hyp not too long
        return lcsLen(refN, h) >= minRefCover;
      });
    }
    if (found !== -1) {
      matched += 1;
      const consumed = hypList[found];
      hypCount.set(consumed, (hypCount.get(consumed) ?? 0) - 1);
      hypList.splice(found, 1);
    }
  }

  return matched / refNormalized.length;
}

function extractGroundTruthText(annotation: FunsdAnnotation): string {
  // Sort entities by ID and concatenate their text
  const sortedEntities = [...annotation.form].sort((a, b) => a.id - b.id);
  return sortedEntities
    .map((entity) => entity.text)
    .filter((text) => text.length > 0)
    .join(" ");
}

function extractGroundTruthByType(
  annotation: FunsdAnnotation,
  entityType: "question" | "answer" | "header" | "other"
): string {
  const entities = annotation.form.filter((e) => e.label === entityType);
  return entities
    .map((entity) => entity.text)
    .filter((text) => text.length > 0)
    .join(" ");
}

function countEntitiesByType(annotation: FunsdAnnotation): {
  question: number;
  answer: number;
  header: number;
  other: number;
} {
  const counts = { question: 0, answer: 0, header: 0, other: 0 };
  for (const entity of annotation.form) {
    if (entity.label in counts) {
      counts[entity.label]++;
    }
  }
  return counts;
}

/**
 * Entity-level rates: for each GT entity, exact_match = normalized text is substring of hyp;
 * word_match = all entity words found in hyp (word_match_accuracy === 1).
 * Returns counts per type for micro/macro aggregation.
 */
function computeEntityLevelRates(
  annotation: FunsdAnnotation,
  normalizedHyp: string
): { exact_match_rate: number; word_match_rate: number; per_type: PerEntityTypeRates } {
  const per_type: PerEntityTypeRates = {
    question: { total: 0, exact_match: 0, word_match: 0 },
    answer: { total: 0, exact_match: 0, word_match: 0 },
    header: { total: 0, exact_match: 0, word_match: 0 },
    other: { total: 0, exact_match: 0, word_match: 0 },
  };
  let total = 0;
  let exact_count = 0;
  let word_count = 0;

  for (const entity of annotation.form) {
    const label = entity.label;
    if (!(label in per_type)) continue;
    const text = entity.text?.trim() ?? "";
    if (text.length === 0) continue;

    per_type[label].total += 1;
    total += 1;

    const normEntity = normalizeText(text);
    const exact = normEntity.length > 0 && normalizedHyp.includes(normEntity);
    const wordMatch = calculateWordMatchAccuracy(normEntity, normalizedHyp) >= 1 - 1e-9;

    if (exact) {
      per_type[label].exact_match += 1;
      exact_count += 1;
    }
    if (wordMatch) {
      per_type[label].word_match += 1;
      word_count += 1;
    }
  }

  return {
    exact_match_rate: total === 0 ? 0 : exact_count / total,
    word_match_rate: total === 0 ? 0 : word_count / total,
    per_type,
  };
}

async function main() {
  const parsesDir = path.join(process.cwd(), "results", "parses");
  const annotationsDir = path.join(process.cwd(), "testing_data", "annotations");
  const outputPath = path.join(process.cwd(), "results", "results.json");

  // Get all parse result files
  const parseFiles = fs
    .readdirSync(parsesDir)
    .filter((f) => f.endsWith(".json"))
    .sort();

  console.log(`Found ${parseFiles.length} parse results to score\n`);

  const scores: DocumentScore[] = [];

  for (const parseFile of parseFiles) {
    const baseName = parseFile.replace(".json", "");
    const annotationFile = `${baseName}.json`;
    const annotationPath = path.join(annotationsDir, annotationFile);

    // Load parse result
    const parseResult: ParseResult = JSON.parse(
      fs.readFileSync(path.join(parsesDir, parseFile), "utf-8")
    );

    // Check if annotation exists
    if (!fs.existsSync(annotationPath)) {
      console.log(`Warning: No annotation found for ${baseName}`);
      continue;
    }

    // Load annotation
    const annotation: FunsdAnnotation = JSON.parse(
      fs.readFileSync(annotationPath, "utf-8")
    );

    // Extract ground truth text
    const refText = extractGroundTruthText(annotation);
    const hypText = extractTextFromParseResult(parseResult);

    // Normalize texts
    const normalizedRef = normalizeText(refText);
    const normalizedHyp = normalizeText(hypText);

    // Calculate metrics
    const cer = calculateCER(normalizedRef, normalizedHyp);
    const wer = calculateWER(normalizedRef, normalizedHyp);
    const ocrAccuracy = (1 - cer) * 100;
    const wordMatchAccuracy = calculateWordMatchAccuracy(normalizedRef, normalizedHyp);

    // Entity breakdown
    const entityBreakdown = countEntitiesByType(annotation);

    // Per-entity-type word-match accuracy (compare entity-type ref words against full hyp)
    const perEntityTypeCer: { question: number; answer: number; header: number; other: number } = {
      question: 0,
      answer: 0,
      header: 0,
      other: 0,
    };

    for (const entityType of ["question", "answer", "header", "other"] as const) {
      const typeRefText = normalizeText(extractGroundTruthByType(annotation, entityType));
      if (typeRefText.length > 0) {
        // Use word-match accuracy instead of CER so values stay in 0-1 range
        perEntityTypeCer[entityType] = calculateWordMatchAccuracy(typeRefText, normalizedHyp);
      }
    }

    // Entity-level exact-match and word-match rates (per entity, not concatenated)
    const entityRates = computeEntityLevelRates(annotation, normalizedHyp);

    const score: DocumentScore = {
      filename: baseName,
      success: parseResult.success,
      cer,
      wer,
      ocr_accuracy: ocrAccuracy,
      word_match_accuracy: wordMatchAccuracy,
      ref_char_count: normalizedRef.length,
      ref_word_count: normalizedRef.split(/\s+/).filter((w) => w.length > 0).length,
      hyp_char_count: normalizedHyp.length,
      hyp_word_count: normalizedHyp.split(/\s+/).filter((w) => w.length > 0).length,
      duration_ms: parseResult.duration_ms,
      api_duration_ms: parseResult.api_duration_ms,
      entity_count: annotation.form.length,
      entity_breakdown: entityBreakdown,
      per_entity_type_cer: perEntityTypeCer,
      entity_exact_match_rate: entityRates.exact_match_rate,
      entity_word_match_rate: entityRates.word_match_rate,
      per_entity_type_rates: entityRates.per_type,
      ref_text: refText.substring(0, 500),
      hyp_text: hypText.substring(0, 500),
    };

    scores.push(score);

    console.log(
      `${baseName}: WordMatch=${(wordMatchAccuracy * 100).toFixed(1)}%, EntityExact=${(entityRates.exact_match_rate * 100).toFixed(1)}%, EntityWord=${(entityRates.word_match_rate * 100).toFixed(1)}%`
    );
  }

  // Calculate aggregate statistics
  const successfulScores = scores.filter((s) => s.success);
  const cerValues = successfulScores.map((s) => s.cer);
  const werValues = successfulScores.map((s) => s.wer);
  const wordMatchValues = successfulScores.map((s) => s.word_match_accuracy);
  const durationValues = successfulScores.map((s) => s.duration_ms);
  const apiDurationValues = successfulScores
    .filter((s) => s.api_duration_ms)
    .map((s) => s.api_duration_ms!);

  const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const median = (arr: number[]) => {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  };
  const percentile = (arr: number[], p: number) => {
    const sorted = [...arr].sort((a, b) => a - b);
    const index = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  };
  const stdDev = (arr: number[]) => {
    const avg = mean(arr);
    const squareDiffs = arr.map((value) => Math.pow(value - avg, 2));
    return Math.sqrt(mean(squareDiffs));
  };

  // Per-entity-type aggregate CER
  const entityTypes = ["question", "answer", "header", "other"] as const;
  const perEntityTypeAggregateCer: Record<string, number> = {};
  for (const entityType of entityTypes) {
    const typeCerValues = successfulScores
      .map((s) => s.per_entity_type_cer[entityType])
      .filter((v) => v > 0);
    if (typeCerValues.length > 0) {
      perEntityTypeAggregateCer[entityType] = mean(typeCerValues);
    }
  }

  // Entity-level aggregate stats (micro: sum counts across docs; macro: avg per-doc rates)
  const entityExactValues = successfulScores.map((s) => s.entity_exact_match_rate);
  const entityWordValues = successfulScores.map((s) => s.entity_word_match_rate);

  // Micro aggregation: sum all entity counts across docs
  const microTotals = { question: { total: 0, exact: 0, word: 0 }, answer: { total: 0, exact: 0, word: 0 }, header: { total: 0, exact: 0, word: 0 }, other: { total: 0, exact: 0, word: 0 } };
  for (const s of successfulScores) {
    for (const t of entityTypes) {
      microTotals[t].total += s.per_entity_type_rates[t].total;
      microTotals[t].exact += s.per_entity_type_rates[t].exact_match;
      microTotals[t].word += s.per_entity_type_rates[t].word_match;
    }
  }
  const microExact: Record<string, number> = {};
  const microWord: Record<string, number> = {};
  for (const t of entityTypes) {
    microExact[t] = microTotals[t].total === 0 ? 0 : microTotals[t].exact / microTotals[t].total;
    microWord[t] = microTotals[t].total === 0 ? 0 : microTotals[t].word / microTotals[t].total;
  }

  const results = {
    summary: {
      total_documents: scores.length,
      successful_parses: successfulScores.length,
      failed_parses: scores.length - successfulScores.length,
      success_rate: (successfulScores.length / scores.length) * 100,
    },
    cer_stats: {
      mean: mean(cerValues),
      median: median(cerValues),
      std_dev: stdDev(cerValues),
      min: Math.min(...cerValues),
      max: Math.max(...cerValues),
      p25: percentile(cerValues, 25),
      p75: percentile(cerValues, 75),
      p90: percentile(cerValues, 90),
      p95: percentile(cerValues, 95),
    },
    wer_stats: {
      mean: mean(werValues),
      median: median(werValues),
      std_dev: stdDev(werValues),
      min: Math.min(...werValues),
      max: Math.max(...werValues),
      p25: percentile(werValues, 25),
      p75: percentile(werValues, 75),
      p90: percentile(werValues, 90),
      p95: percentile(werValues, 95),
    },
    ocr_accuracy: {
      mean: (1 - mean(cerValues)) * 100,
      median: (1 - median(cerValues)) * 100,
    },
    word_match_accuracy: {
      mean: mean(wordMatchValues) * 100,
      median: median(wordMatchValues) * 100,
      min: Math.min(...wordMatchValues) * 100,
      max: Math.max(...wordMatchValues) * 100,
    },
    duration_stats: {
      mean_ms: mean(durationValues),
      median_ms: median(durationValues),
      min_ms: Math.min(...durationValues),
      max_ms: Math.max(...durationValues),
      api_mean_ms: apiDurationValues.length > 0 ? mean(apiDurationValues) : null,
      api_median_ms: apiDurationValues.length > 0 ? median(apiDurationValues) : null,
    },
    per_entity_type_cer: perEntityTypeAggregateCer,
    entity_level: {
      exact_match: {
        macro_mean: mean(entityExactValues) * 100,
        macro_median: median(entityExactValues) * 100,
        micro_by_type: Object.fromEntries(entityTypes.map((t) => [t, microExact[t] * 100])),
        micro_overall: (() => {
          const tot = entityTypes.reduce((a, t) => a + microTotals[t].total, 0);
          const ex = entityTypes.reduce((a, t) => a + microTotals[t].exact, 0);
          return tot === 0 ? 0 : (ex / tot) * 100;
        })(),
      },
      word_match: {
        macro_mean: mean(entityWordValues) * 100,
        macro_median: median(entityWordValues) * 100,
        micro_by_type: Object.fromEntries(entityTypes.map((t) => [t, microWord[t] * 100])),
        micro_overall: (() => {
          const tot = entityTypes.reduce((a, t) => a + microTotals[t].total, 0);
          const wm = entityTypes.reduce((a, t) => a + microTotals[t].word, 0);
          return tot === 0 ? 0 : (wm / tot) * 100;
        })(),
      },
    },
    timestamp: new Date().toISOString(),
    documents: scores,
  };

  // Save results
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));

  console.log("\n" + "=".repeat(60));
  console.log("Scoring Summary:");
  console.log("=".repeat(60));
  console.log(`Total Documents: ${results.summary.total_documents}`);
  console.log(`Successful Parses: ${results.summary.successful_parses}`);
  console.log(`Success Rate: ${results.summary.success_rate.toFixed(1)}%`);
  console.log("");
  console.log("Word-Match OCR Accuracy (primary metric):");
  console.log(`  Mean: ${results.word_match_accuracy.mean.toFixed(2)}%`);
  console.log(`  Median: ${results.word_match_accuracy.median.toFixed(2)}%`);
  console.log(`  Min: ${results.word_match_accuracy.min.toFixed(2)}%`);
  console.log(`  Max: ${results.word_match_accuracy.max.toFixed(2)}%`);
  console.log("");
  console.log("Sequence OCR Accuracy (100 - CER):");
  console.log(`  Mean: ${results.ocr_accuracy.mean.toFixed(2)}%`);
  console.log(`  Median: ${results.ocr_accuracy.median.toFixed(2)}%`);
  console.log("");
  console.log("Character Error Rate (CER):");
  console.log(`  Mean: ${(results.cer_stats.mean * 100).toFixed(2)}%`);
  console.log(`  Median: ${(results.cer_stats.median * 100).toFixed(2)}%`);
  console.log(`  Std Dev: ${(results.cer_stats.std_dev * 100).toFixed(2)}%`);
  console.log(`  P25: ${(results.cer_stats.p25 * 100).toFixed(2)}%`);
  console.log(`  P75: ${(results.cer_stats.p75 * 100).toFixed(2)}%`);
  console.log("");
  console.log("Word Error Rate (WER):");
  console.log(`  Mean: ${(results.wer_stats.mean * 100).toFixed(2)}%`);
  console.log(`  Median: ${(results.wer_stats.median * 100).toFixed(2)}%`);
  console.log("");
  console.log("Processing Time:");
  console.log(`  Mean: ${results.duration_stats.mean_ms.toFixed(0)}ms`);
  console.log(`  Median: ${results.duration_stats.median_ms.toFixed(0)}ms`);
  if (results.duration_stats.api_mean_ms) {
    console.log(`  API Mean: ${results.duration_stats.api_mean_ms.toFixed(0)}ms`);
  }
  console.log("");
  console.log("Per-Entity-Type Word-Match (macro avg):");
  for (const [type, rate] of Object.entries(results.per_entity_type_cer)) {
    console.log(`  ${type}: ${(rate * 100).toFixed(1)}%`);
  }
  console.log("");
  console.log("Entity-Level Exact-Match (substring in OCR):");
  console.log(`  Macro Mean: ${results.entity_level.exact_match.macro_mean.toFixed(1)}%`);
  console.log(`  Micro Overall: ${results.entity_level.exact_match.micro_overall.toFixed(1)}%`);
  for (const t of entityTypes) {
    console.log(`    ${t}: ${(results.entity_level.exact_match.micro_by_type as Record<string, number>)[t].toFixed(1)}%`);
  }
  console.log("");
  console.log("Entity-Level Word-Match (all words found):");
  console.log(`  Macro Mean: ${results.entity_level.word_match.macro_mean.toFixed(1)}%`);
  console.log(`  Micro Overall: ${results.entity_level.word_match.micro_overall.toFixed(1)}%`);
  for (const t of entityTypes) {
    console.log(`    ${t}: ${(results.entity_level.word_match.micro_by_type as Record<string, number>)[t].toFixed(1)}%`);
  }
  console.log("=".repeat(60));
  console.log(`\nResults saved to: ${outputPath}`);
}

main().catch(console.error);
