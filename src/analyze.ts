import * as fs from "fs";
import * as path from "path";
import { distance } from "fastest-levenshtein";

interface DocumentScore {
  filename: string;
  success: boolean;
  cer: number;
  wer: number;
  ocr_accuracy: number;
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

interface Results {
  summary: {
    total_documents: number;
    successful_parses: number;
    failed_parses: number;
    success_rate: number;
  };
  cer_stats: {
    mean: number;
    median: number;
    std_dev: number;
    min: number;
    max: number;
    p25: number;
    p75: number;
    p90: number;
    p95: number;
  };
  wer_stats: {
    mean: number;
    median: number;
    std_dev: number;
    min: number;
    max: number;
    p25: number;
    p75: number;
    p90: number;
    p95: number;
  };
  ocr_accuracy: {
    mean: number;
    median: number;
  };
  duration_stats: {
    mean_ms: number;
    median_ms: number;
    min_ms: number;
    max_ms: number;
    api_mean_ms: number | null;
    api_median_ms: number | null;
  };
  per_entity_type_cer: Record<string, number>;
  timestamp: string;
  documents: DocumentScore[];
}

interface ErrorPattern {
  original: string;
  recognized: string;
  count: number;
  type: "substitution" | "insertion" | "deletion";
}

function analyzeErrorPatterns(refText: string, hypText: string): ErrorPattern[] {
  const patterns: Map<string, ErrorPattern> = new Map();

  // Simple character-level error analysis
  const refChars = refText.split("");
  const hypChars = hypText.split("");

  // Use dynamic programming to find alignment
  const m = refChars.length;
  const n = hypChars.length;

  // Build DP table
  const dp: number[][] = Array(m + 1)
    .fill(null)
    .map(() => Array(n + 1).fill(0));

  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (refChars[i - 1] === hypChars[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      }
    }
  }

  // Backtrack to find operations
  let i = m;
  let j = n;

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && refChars[i - 1] === hypChars[j - 1]) {
      i--;
      j--;
    } else if (i > 0 && j > 0 && dp[i][j] === dp[i - 1][j - 1] + 1) {
      // Substitution
      const key = `sub:${refChars[i - 1]}:${hypChars[j - 1]}`;
      const existing = patterns.get(key);
      if (existing) {
        existing.count++;
      } else {
        patterns.set(key, {
          original: refChars[i - 1],
          recognized: hypChars[j - 1],
          count: 1,
          type: "substitution",
        });
      }
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j] === dp[i][j - 1] + 1)) {
      // Insertion
      const key = `ins:${hypChars[j - 1]}`;
      const existing = patterns.get(key);
      if (existing) {
        existing.count++;
      } else {
        patterns.set(key, {
          original: "",
          recognized: hypChars[j - 1],
          count: 1,
          type: "insertion",
        });
      }
      j--;
    } else if (i > 0) {
      // Deletion
      const key = `del:${refChars[i - 1]}`;
      const existing = patterns.get(key);
      if (existing) {
        existing.count++;
      } else {
        patterns.set(key, {
          original: refChars[i - 1],
          recognized: "",
          count: 1,
          type: "deletion",
        });
      }
      i--;
    }
  }

  return Array.from(patterns.values()).sort((a, b) => b.count - a.count);
}

function calculateCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n !== y.length || n === 0) return 0;

  const meanX = x.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;

  let numerator = 0;
  let denomX = 0;
  let denomY = 0;

  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    numerator += dx * dy;
    denomX += dx * dx;
    denomY += dy * dy;
  }

  const denominator = Math.sqrt(denomX * denomY);
  return denominator === 0 ? 0 : numerator / denominator;
}

async function main() {
  const resultsPath = path.join(process.cwd(), "results", "results.json");
  const analysisPath = path.join(process.cwd(), "results", "analysis.json");

  if (!fs.existsSync(resultsPath)) {
    console.error("Error: results.json not found. Run 'npm run score' first.");
    process.exit(1);
  }

  const results: Results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
  const docs = results.documents.filter((d) => d.success);

  console.log("Performing deep analysis on", docs.length, "documents...\n");

  // 1. Complexity correlations
  const entityCounts = docs.map((d) => d.entity_count);
  const wordCounts = docs.map((d) => d.ref_word_count);
  const charCounts = docs.map((d) => d.ref_char_count);
  const cerValues = docs.map((d) => d.cer);
  const werValues = docs.map((d) => d.wer);
  const durations = docs.map((d) => d.duration_ms);

  const correlations = {
    entity_count_vs_cer: calculateCorrelation(entityCounts, cerValues),
    word_count_vs_cer: calculateCorrelation(wordCounts, cerValues),
    char_count_vs_cer: calculateCorrelation(charCounts, cerValues),
    entity_count_vs_wer: calculateCorrelation(entityCounts, werValues),
    word_count_vs_wer: calculateCorrelation(wordCounts, werValues),
    cer_vs_wer: calculateCorrelation(cerValues, werValues),
    char_count_vs_duration: calculateCorrelation(charCounts, durations),
    word_count_vs_duration: calculateCorrelation(wordCounts, durations),
  };

  // 2. Error pattern analysis (aggregate across all documents)
  const allErrorPatterns: Map<string, ErrorPattern> = new Map();

  for (const doc of docs) {
    if (doc.ref_text && doc.hyp_text) {
      const patterns = analyzeErrorPatterns(
        doc.ref_text.toLowerCase(),
        doc.hyp_text.toLowerCase()
      );
      for (const pattern of patterns) {
        const key = `${pattern.type}:${pattern.original}:${pattern.recognized}`;
        const existing = allErrorPatterns.get(key);
        if (existing) {
          existing.count += pattern.count;
        } else {
          allErrorPatterns.set(key, { ...pattern });
        }
      }
    }
  }

  const topErrorPatterns = Array.from(allErrorPatterns.values())
    .sort((a, b) => b.count - a.count)
    .slice(0, 50);

  // 3. CER distribution buckets
  const cerBuckets = {
    "0-5%": docs.filter((d) => d.cer < 0.05).length,
    "5-10%": docs.filter((d) => d.cer >= 0.05 && d.cer < 0.1).length,
    "10-15%": docs.filter((d) => d.cer >= 0.1 && d.cer < 0.15).length,
    "15-20%": docs.filter((d) => d.cer >= 0.15 && d.cer < 0.2).length,
    "20-25%": docs.filter((d) => d.cer >= 0.2 && d.cer < 0.25).length,
    "25%+": docs.filter((d) => d.cer >= 0.25).length,
  };

  // 4. WER distribution buckets
  const werBuckets = {
    "0-20%": docs.filter((d) => d.wer < 0.2).length,
    "20-40%": docs.filter((d) => d.wer >= 0.2 && d.wer < 0.4).length,
    "40-60%": docs.filter((d) => d.wer >= 0.4 && d.wer < 0.6).length,
    "60-80%": docs.filter((d) => d.wer >= 0.6 && d.wer < 0.8).length,
    "80-100%": docs.filter((d) => d.wer >= 0.8).length,
  };

  // 5. Duration distribution buckets
  const durationBuckets = {
    "0-1s": docs.filter((d) => d.duration_ms < 1000).length,
    "1-2s": docs.filter((d) => d.duration_ms >= 1000 && d.duration_ms < 2000).length,
    "2-3s": docs.filter((d) => d.duration_ms >= 2000 && d.duration_ms < 3000).length,
    "3-5s": docs.filter((d) => d.duration_ms >= 3000 && d.duration_ms < 5000).length,
    "5s+": docs.filter((d) => d.duration_ms >= 5000).length,
  };

  // 6. Best and worst performing documents
  const sortedByAccuracy = [...docs].sort((a, b) => b.word_match_accuracy - a.word_match_accuracy);
  const bestDocs = sortedByAccuracy.slice(0, 5).map((d) => ({
    filename: d.filename,
    cer: d.cer,
    wer: d.wer,
    ocr_accuracy: d.ocr_accuracy,
    word_match_accuracy: d.word_match_accuracy,
    entity_count: d.entity_count,
  }));
  const worstDocs = sortedByAccuracy.slice(-5).reverse().map((d) => ({
    filename: d.filename,
    cer: d.cer,
    wer: d.wer,
    ocr_accuracy: d.ocr_accuracy,
    word_match_accuracy: d.word_match_accuracy,
    entity_count: d.entity_count,
  }));

  // 7. Entity type breakdown statistics
  const entityTypeStats: Record<string, { total: number; avg_per_doc: number }> = {};
  const entityTypes = ["question", "answer", "header", "other"] as const;

  for (const type of entityTypes) {
    const counts = docs.map((d) => d.entity_breakdown[type]);
    const total = counts.reduce((a, b) => a + b, 0);
    entityTypeStats[type] = {
      total,
      avg_per_doc: total / docs.length,
    };
  }

  // 8. Cumulative distribution function data for CER
  const sortedCer = [...cerValues].sort((a, b) => a - b);
  const cdfData = sortedCer.map((cer, i) => ({
    cer,
    cumulative_pct: ((i + 1) / sortedCer.length) * 100,
  }));

  // 9. Scatter plot data for visualizations
  const scatterData = docs.map((d) => ({
    filename: d.filename,
    cer: d.cer,
    wer: d.wer,
    entity_count: d.entity_count,
    word_count: d.ref_word_count,
    char_count: d.ref_char_count,
    duration_ms: d.duration_ms,
    ocr_accuracy: d.ocr_accuracy,
    word_match_accuracy: d.word_match_accuracy,
  }));

  const analysis = {
    correlations,
    cer_buckets: cerBuckets,
    wer_buckets: werBuckets,
    duration_buckets: durationBuckets,
    top_error_patterns: topErrorPatterns,
    best_documents: bestDocs,
    worst_documents: worstDocs,
    entity_type_stats: entityTypeStats,
    cdf_data: cdfData,
    scatter_data: scatterData,
    timestamp: new Date().toISOString(),
  };

  fs.writeFileSync(analysisPath, JSON.stringify(analysis, null, 2));

  console.log("=".repeat(60));
  console.log("Deep Analysis Results");
  console.log("=".repeat(60));

  console.log("\nCorrelations:");
  console.log(`  Entity Count vs CER: ${correlations.entity_count_vs_cer.toFixed(3)}`);
  console.log(`  Word Count vs CER: ${correlations.word_count_vs_cer.toFixed(3)}`);
  console.log(`  Char Count vs CER: ${correlations.char_count_vs_cer.toFixed(3)}`);
  console.log(`  CER vs WER: ${correlations.cer_vs_wer.toFixed(3)}`);
  console.log(`  Char Count vs Duration: ${correlations.char_count_vs_duration.toFixed(3)}`);

  console.log("\nCER Distribution:");
  for (const [bucket, count] of Object.entries(cerBuckets)) {
    const pct = ((count / docs.length) * 100).toFixed(1);
    console.log(`  ${bucket}: ${count} docs (${pct}%)`);
  }

  console.log("\nWER Distribution:");
  for (const [bucket, count] of Object.entries(werBuckets)) {
    const pct = ((count / docs.length) * 100).toFixed(1);
    console.log(`  ${bucket}: ${count} docs (${pct}%)`);
  }

  console.log("\nDuration Distribution:");
  for (const [bucket, count] of Object.entries(durationBuckets)) {
    const pct = ((count / docs.length) * 100).toFixed(1);
    console.log(`  ${bucket}: ${count} docs (${pct}%)`);
  }

  console.log("\nTop 10 Error Patterns:");
  for (const pattern of topErrorPatterns.slice(0, 10)) {
    const orig = pattern.original || "(none)";
    const rec = pattern.recognized || "(none)";
    console.log(`  ${pattern.type}: "${orig}" → "${rec}" (${pattern.count}x)`);
  }

  console.log("\nBest Performing Documents (highest word-match accuracy):");
  for (const doc of bestDocs) {
    console.log(
      `  ${doc.filename}: WordMatch=${(doc.word_match_accuracy * 100).toFixed(2)}%, CER=${(doc.cer * 100).toFixed(2)}%`
    );
  }

  console.log("\nWorst Performing Documents (lowest word-match accuracy):");
  for (const doc of worstDocs) {
    console.log(
      `  ${doc.filename}: WordMatch=${(doc.word_match_accuracy * 100).toFixed(2)}%, CER=${(doc.cer * 100).toFixed(2)}%`
    );
  }

  console.log("\nEntity Type Statistics:");
  for (const [type, stats] of Object.entries(entityTypeStats)) {
    console.log(`  ${type}: ${stats.total} total (${stats.avg_per_doc.toFixed(1)} avg/doc)`);
  }

  console.log("=".repeat(60));
  console.log(`\nAnalysis saved to: ${analysisPath}`);
}

main().catch(console.error);
