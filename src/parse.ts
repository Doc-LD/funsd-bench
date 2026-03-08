import * as fs from "fs";
import * as path from "path";
import "dotenv/config";

const API_KEY = process.env.DOCLD_API_KEY;
const BASE_URL = process.env.DOCLD_BASE_URL || "http://localhost:3000";

if (!API_KEY) {
  console.error("Error: DOCLD_API_KEY environment variable is required");
  process.exit(1);
}

interface ParseResponse {
  job_id: string;
  duration: number;
  usage?: {
    num_pages: number;
    credits: number;
  };
  result: {
    type: string;
    chunks: Array<{
      content: string;
      blocks: Array<{
        type: string;
        content: string;
        bbox?: {
          page: number;
          left: number;
          top: number;
          width: number;
          height: number;
        };
        confidence?: string;
      }>;
    }>;
  };
  studio_link?: string;
}

interface ParseResult {
  filename: string;
  success: boolean;
  duration_ms: number;
  api_duration_ms?: number;
  error?: string;
  response?: { data?: ParseResponse; meta?: unknown } | ParseResponse;
  extracted_text?: string;
}

async function parseImage(imagePath: string): Promise<ParseResult> {
  const filename = path.basename(imagePath);
  const startTime = Date.now();

  try {
    // Read file as buffer and create FormData using native API
    const fileBuffer = fs.readFileSync(imagePath);
    const blob = new Blob([fileBuffer], { type: "image/png" });
    
    const formData = new FormData();
    formData.append("file", blob, filename);

    const response = await fetch(`${BASE_URL}/api/parse`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${API_KEY}`,
      },
      body: formData,
    });

    const duration_ms = Date.now() - startTime;

    if (!response.ok) {
      const errorText = await response.text();
      return {
        filename,
        success: false,
        duration_ms,
        error: `HTTP ${response.status}: ${errorText}`,
      };
    }

    const data = (await response.json()) as { data?: ParseResponse; meta?: unknown };

    // Handle both direct response and wrapped response
    const parseResponse = data.data || (data as unknown as ParseResponse);

    // Extract all text from blocks
    let extractedText = "";
    if (parseResponse.result?.chunks) {
      for (const chunk of parseResponse.result.chunks) {
        if (chunk.blocks) {
          for (const block of chunk.blocks) {
            if (block.content) {
              extractedText += block.content + " ";
            }
          }
        }
        // Also try chunk.content directly
        if (chunk.content && !chunk.blocks?.length) {
          extractedText += chunk.content + " ";
        }
      }
    }

    return {
      filename,
      success: true,
      duration_ms,
      api_duration_ms: parseResponse.duration ? parseResponse.duration * 1000 : undefined,
      response: data,
      extracted_text: extractedText.trim(),
    };
  } catch (error) {
    const duration_ms = Date.now() - startTime;
    return {
      filename,
      success: false,
      duration_ms,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function parseImageWithRetry(
  imagePath: string,
  maxRetries: number = 3,
  baseDelay: number = 5000
): Promise<ParseResult> {
  let lastError: string | undefined;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (attempt > 0) {
      const delay = baseDelay * Math.pow(2, attempt - 1);
      console.log(`    Retry ${attempt}/${maxRetries} after ${delay}ms...`);
      await sleep(delay);
    }

    const result = await parseImage(imagePath);

    if (result.success) {
      return result;
    }

    lastError = result.error;

    // Don't retry on non-transient errors
    if (
      result.error &&
      !result.error.includes("500") &&
      !result.error.includes("502") &&
      !result.error.includes("503") &&
      !result.error.includes("504") &&
      !result.error.includes("Circuit breaker") &&
      !result.error.includes("ECONNRESET") &&
      !result.error.includes("ETIMEDOUT")
    ) {
      return result;
    }
  }

  return {
    filename: path.basename(imagePath),
    success: false,
    duration_ms: 0,
    error: lastError || "Max retries exceeded",
  };
}

async function main() {
  const imagesDir = path.join(process.cwd(), "testing_data", "images");
  const outputDir = path.join(process.cwd(), "results", "parses");

  // Ensure output directory exists
  fs.mkdirSync(outputDir, { recursive: true });

  // Get all image files
  const imageFiles = fs
    .readdirSync(imagesDir)
    .filter((f) => f.endsWith(".png") || f.endsWith(".jpg"))
    .sort();

  console.log(`Found ${imageFiles.length} images to process`);
  console.log(`API Base URL: ${BASE_URL}`);
  console.log(`Output directory: ${outputDir}\n`);

  const results: ParseResult[] = [];
  let successCount = 0;
  let failCount = 0;

  // Check for --skip-existing flag
  const skipExisting = process.argv.includes("--skip-existing");

  // Check for --limit flag
  const limitIndex = process.argv.indexOf("--limit");
  const limit =
    limitIndex !== -1 ? parseInt(process.argv[limitIndex + 1]) : imageFiles.length;

  const filesToProcess = imageFiles.slice(0, limit);

  for (let i = 0; i < filesToProcess.length; i++) {
    const imageFile = filesToProcess[i];
    const imagePath = path.join(imagesDir, imageFile);
    const outputPath = path.join(
      outputDir,
      imageFile.replace(/\.(png|jpg)$/, ".json")
    );

    // Skip if already processed successfully
    if (skipExisting && fs.existsSync(outputPath)) {
      const existingResult = JSON.parse(fs.readFileSync(outputPath, "utf-8"));
      if (existingResult.success) {
        console.log(`[${i + 1}/${filesToProcess.length}] Skipping ${imageFile} (already successful)`);
        results.push(existingResult);
        successCount++;
        continue;
      }
      // Re-process failed ones
      console.log(`[${i + 1}/${filesToProcess.length}] Re-processing ${imageFile} (previous attempt failed)`);
    } else {
      console.log(`[${i + 1}/${filesToProcess.length}] Processing ${imageFile}...`);
    }

    const result = await parseImageWithRetry(imagePath, 3, 10000);
    results.push(result);

    // Save individual result
    fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));

    if (result.success) {
      successCount++;
      console.log(
        `  ✓ Success (${result.duration_ms}ms, API: ${result.api_duration_ms?.toFixed(0) || "N/A"}ms)`
      );
    } else {
      failCount++;
      console.log(`  ✗ Failed: ${result.error}`);
    }

    // Minimal delay between requests (OpenAI supports high throughput)
    if (i < filesToProcess.length - 1) {
      await sleep(200);
    }
  }

  // Save summary
  const summary = {
    total: results.length,
    success: successCount,
    failed: failCount,
    success_rate: (successCount / results.length) * 100,
    avg_duration_ms:
      results
        .filter((r) => r.success)
        .reduce((sum, r) => sum + r.duration_ms, 0) / successCount,
    avg_api_duration_ms:
      results
        .filter((r) => r.success && r.api_duration_ms)
        .reduce((sum, r) => sum + (r.api_duration_ms || 0), 0) /
      results.filter((r) => r.success && r.api_duration_ms).length,
    timestamp: new Date().toISOString(),
  };

  const summaryPath = path.join(process.cwd(), "results", "parse_summary.json");
  fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));

  console.log("\n" + "=".repeat(50));
  console.log("Parse Summary:");
  console.log(`  Total: ${summary.total}`);
  console.log(`  Success: ${summary.success} (${summary.success_rate.toFixed(1)}%)`);
  console.log(`  Failed: ${summary.failed}`);
  console.log(`  Avg Duration: ${summary.avg_duration_ms.toFixed(0)}ms`);
  console.log(`  Avg API Duration: ${summary.avg_api_duration_ms.toFixed(0)}ms`);
  console.log("=".repeat(50));
}

main().catch(console.error);
