# FUNSD benchmark for DocLD

Benchmark for [DocLD](https://docld.com) form understanding on the [FUNSD](https://guillaumejaume.github.io/FUNSD/) test set (50 images).

## Quick start

```bash
cp .env.example .env   # set DOCLD_API_KEY
npm install
npm run parse          # parse all 50 test images via DocLD API
npm run score          # compute word-match accuracy, CER, WER
npm run analyze        # statistical analysis
npm run charts         # generate visualizations (requires Python + deps in scripts/requirements.txt)
```

## Data

- **Test set:** `testing_data/images/` (50 PNGs) and `testing_data/annotations/` (50 JSON ground-truth files).
- Same data is available as [Hugging Face dataset `docld/funsd-bench`](https://huggingface.co/datasets/docld/funsd-bench).

## Results

See [DocLD blog: FUNSD performance analysis](https://docld.com/blog/docld-funsd) for full methodology and charts.

## License

MIT. FUNSD dataset is for research use; see [FUNSD](https://guillaumejaume.github.io/FUNSD/).
