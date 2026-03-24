/**
 * GST Calculation API
 * Runs at http://localhost:3000
 * 
 * Endpoints:
 *   POST /gst-calculation       → calculate GST
 *   POST /reverse-gst           → reverse GST calculation
 *   POST /gst-breakdown         → CGST/SGST/IGST breakdown
 *   POST /compare-rates         → compare multiple GST rates
 *   GET  /health                → health check
 */

const express = require("express");
const app = express();
app.use(express.json());

const PORT = 3000;

// ── Helpers ────────────────────────────────────────────────────────────────────

function validatePositive(value, name) {
  if (typeof value !== "number" || value < 0) {
    throw new Error(`${name} must be a positive number`);
  }
}

// ── POST /gst-calculation ──────────────────────────────────────────────────────
app.post("/gst-calculation", (req, res) => {
  try {
    const { base_amount, gst_rate } = req.body;

    if (base_amount === undefined || gst_rate === undefined) {
      return res.status(400).json({
        error: "Missing required fields: base_amount, gst_rate"
      });
    }

    validatePositive(base_amount, "base_amount");
    validatePositive(gst_rate, "gst_rate");

    const gst_amount   = parseFloat(((base_amount * gst_rate) / 100).toFixed(2));
    const total_amount = parseFloat((base_amount + gst_amount).toFixed(2));

    console.log(`[calculate_gst] base=${base_amount}, rate=${gst_rate}% → gst=${gst_amount}, total=${total_amount}`);

    return res.json({
      base_amount:  parseFloat(base_amount.toFixed(2)),
      gst_rate,
      gst_amount,
      total_amount,
      source: "gst_api"
    });

  } catch (err) {
    console.error("[calculate_gst] Error:", err.message);
    return res.status(400).json({ error: err.message });
  }
});

// ── POST /reverse-gst ─────────────────────────────────────────────────────────
app.post("/reverse-gst", (req, res) => {
  try {
    const { total_amount, gst_rate } = req.body;

    if (total_amount === undefined || gst_rate === undefined) {
      return res.status(400).json({
        error: "Missing required fields: total_amount, gst_rate"
      });
    }

    validatePositive(total_amount, "total_amount");
    validatePositive(gst_rate, "gst_rate");

    const base_amount = parseFloat((total_amount / (1 + gst_rate / 100)).toFixed(2));
    const gst_amount  = parseFloat((total_amount - base_amount).toFixed(2));

    console.log(`[reverse_gst] total=${total_amount}, rate=${gst_rate}% → base=${base_amount}, gst=${gst_amount}`);

    return res.json({
      total_amount: parseFloat(total_amount.toFixed(2)),
      gst_rate,
      base_amount,
      gst_amount,
      source: "gst_api"
    });

  } catch (err) {
    console.error("[reverse_gst] Error:", err.message);
    return res.status(400).json({ error: err.message });
  }
});

// ── POST /gst-breakdown ───────────────────────────────────────────────────────
app.post("/gst-breakdown", (req, res) => {
  try {
    const { base_amount, gst_rate, is_intra_state = true } = req.body;

    if (base_amount === undefined || gst_rate === undefined) {
      return res.status(400).json({
        error: "Missing required fields: base_amount, gst_rate"
      });
    }

    validatePositive(base_amount, "base_amount");
    validatePositive(gst_rate, "gst_rate");

    const gst_amount   = parseFloat(((base_amount * gst_rate) / 100).toFixed(2));
    const total_amount = parseFloat((base_amount + gst_amount).toFixed(2));

    const breakdown = is_intra_state
      ? {
          type:      "Intra-State",
          cgst:      parseFloat((gst_amount / 2).toFixed(2)),
          sgst:      parseFloat((gst_amount / 2).toFixed(2)),
          igst:      0,
          cgst_rate: gst_rate / 2,
          sgst_rate: gst_rate / 2,
          igst_rate: 0
        }
      : {
          type:      "Inter-State",
          cgst:      0,
          sgst:      0,
          igst:      gst_amount,
          cgst_rate: 0,
          sgst_rate: 0,
          igst_rate: gst_rate
        };

    console.log(`[gst_breakdown] base=${base_amount}, rate=${gst_rate}%, intra=${is_intra_state}`);

    return res.json({
      base_amount: parseFloat(base_amount.toFixed(2)),
      gst_rate,
      gst_amount,
      total_amount,
      source: "gst_api",
      breakdown
    });

  } catch (err) {
    console.error("[gst_breakdown] Error:", err.message);
    return res.status(400).json({ error: err.message });
  }
});

// ── POST /compare-rates ───────────────────────────────────────────────────────
app.post("/compare-rates", (req, res) => {
  try {
    const { base_amount, rates } = req.body;

    if (base_amount === undefined || !Array.isArray(rates) || rates.length === 0) {
      return res.status(400).json({
        error: "Missing required fields: base_amount, rates (array)"
      });
    }

    validatePositive(base_amount, "base_amount");

    const comparisons = rates
      .map(rate => {
        const gst_amount   = parseFloat(((base_amount * rate) / 100).toFixed(2));
        const total_amount = parseFloat((base_amount + gst_amount).toFixed(2));
        return { rate, base_amount, gst_rate: rate, gst_amount, total_amount, source: "gst_api" };
      })
      .sort((a, b) => a.rate - b.rate);

    const lowest = comparisons[0].total_amount;
    comparisons.forEach(c => {
      c.difference_from_lowest = parseFloat((c.total_amount - lowest).toFixed(2));
    });

    console.log(`[compare_rates] base=${base_amount}, rates=${rates}`);

    return res.json({
      base_amount,
      comparisons,
      lowest_rate:    comparisons[0].rate,
      highest_rate:   comparisons[comparisons.length - 1].rate,
      max_difference: parseFloat(
        (comparisons[comparisons.length - 1].total_amount - comparisons[0].total_amount).toFixed(2)
      )
    });

  } catch (err) {
    console.error("[compare_rates] Error:", err.message);
    return res.status(400).json({ error: err.message });
  }
});

// ── GET /health ────────────────────────────────────────────────────────────────
app.get("/health", (req, res) => {
  res.json({
    status:    "ok",
    service:   "GST Calculation API",
    version:   "1.0.0",
    timestamp: new Date().toISOString(),
    endpoints: [
      "POST /gst-calculation",
      "POST /reverse-gst",
      "POST /gst-breakdown",
      "POST /compare-rates",
      "GET  /health"
    ]
  });
});

// ── 404 handler ───────────────────────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({
    error:    `Route ${req.method} ${req.path} not found`,
    available: ["POST /gst-calculation", "POST /reverse-gst",
                "POST /gst-breakdown", "POST /compare-rates", "GET /health"]
  });
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`✓ GST API running at http://localhost:${PORT}`);
  console.log(`  POST /gst-calculation`);
  console.log(`  POST /reverse-gst`);
  console.log(`  POST /gst-breakdown`);
  console.log(`  POST /compare-rates`);
  console.log(`  GET  /health`);
});