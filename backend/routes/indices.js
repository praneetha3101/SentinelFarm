const express = require("express");
const axios = require("axios");
const router = express.Router();

// Get Flask backend URL from environment or use default
const FLASK_URL = process.env.FLASK_API_URL || "http://localhost:5001";

/**
 * GET /api/indices/list
 * Fetch available vegetation indices
 */
router.get("/list", async (req, res) => {
  try {
    console.log(`📤 Proxying GET /api/indices/list to Flask backend at ${FLASK_URL}`);
    
    const response = await axios.get(`${FLASK_URL}/api/indices/list`, {
      timeout: 30000
    });

    return res.status(response.status).json(response.data);
  } catch (error) {
    console.error("❌ Error fetching available indices:", error.message);
    
    // If Flask is down, return a default list of available indices
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      console.log("⚠️ Flask backend unavailable, returning default indices list");
      return res.status(200).json({
        status: "success",
        indices: [
          {
            name: "NDVI",
            description: "Normalized Difference Vegetation Index",
            formula: "(NIR - R) / (NIR + R)"
          },
          {
            name: "EVI",
            description: "Enhanced Vegetation Index",
            formula: "2.5 * ((NIR - R) / (NIR + 6 * R - 7.5 * B + 1))"
          },
          {
            name: "SAVI",
            description: "Soil-Adjusted Vegetation Index",
            formula: "((NIR - R) / (NIR + R + 0.5)) * 1.5"
          },
          {
            name: "ARVI",
            description: "Atmospherically Resistant Vegetation Index",
            formula: "(NIR - (2 * R - B)) / (NIR + (2 * R - B))"
          },
          {
            name: "MAVI",
            description: "Modified Adjusted Vegetation Index",
            formula: "(NIR - R) / (NIR + R + SWIR)"
          },
          {
            name: "SR",
            description: "Simple Ratio",
            formula: "NIR / R"
          }
        ],
        note: "Flask backend unavailable - returning cached indices"
      });
    }

    return res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Failed to fetch available indices",
      details: error.message
    });
  }
});

/**
 * POST /api/indices/calculate
 * Calculate a vegetation index for given coordinates and dates
 * 
 * Expected body:
 * {
 *   "coordinates": [[lng, lat], [lng, lat], ...],
 *   "start_date": "YYYY-MM-DD",
 *   "end_date": "YYYY-MM-DD",
 *   "index_name": "NDVI" | "EVI" | "SAVI" | "ARVI" | "MAVI" | "SR"
 * }
 */
router.post("/calculate", async (req, res) => {
  try {
    const { coordinates, start_date, end_date, index_name = "NDVI" } = req.body;

    // Validate input
    if (!coordinates || !start_date || !end_date) {
      return res.status(400).json({
        error: "Missing required fields: coordinates, start_date, end_date"
      });
    }

    console.log(`📤 Proxying POST /api/indices/calculate to Flask backend`);
    console.log(`   Index: ${index_name}, AOI points: ${coordinates.length}`);

    const response = await axios.post(
      `${FLASK_URL}/api/indices/calculate`,
      {
        coordinates,
        start_date,
        end_date,
        index_name
      },
      { timeout: 120000 } // 2 minute timeout for image processing
    );

    return res.status(response.status).json(response.data);
  } catch (error) {
    console.error("❌ Error calculating index:", error.message);

    // Check if error response has details
    const errorData = error.response?.data;
    const statusCode = error.response?.status || 500;

    return res.status(statusCode).json({
      error: errorData?.error || "Failed to calculate vegetation index",
      details: errorData?.details || error.message,
      suggestion: errorData?.suggestion,
      debugging: errorData?.debugging
    });
  }
});

/**
 * POST /api/indices/timeseries
 * Calculate time series for a vegetation index
 * 
 * Expected body:
 * {
 *   "coordinates": [[lng, lat], [lng, lat], ...],
 *   "start_date": "YYYY-MM-DD",
 *   "end_date": "YYYY-MM-DD",
 *   "index_name": "NDVI" | "EVI" | "SAVI" | "ARVI" | "MAVI" | "SR"
 * }
 */
router.post("/timeseries", async (req, res) => {
  try {
    const { coordinates, start_date, end_date, index_name = "NDVI" } = req.body;

    // Validate input
    if (!coordinates || !start_date || !end_date) {
      return res.status(400).json({
        error: "Missing required fields: coordinates, start_date, end_date"
      });
    }

    console.log(`📤 Proxying POST /api/indices/timeseries to Flask backend`);
    console.log(`   Index: ${index_name}, AOI points: ${coordinates.length}`);

    const response = await axios.post(
      `${FLASK_URL}/api/indices/timeseries`,
      {
        coordinates,
        start_date,
        end_date,
        index_name
      },
      { timeout: 180000 } // 3 minute timeout for time series processing
    );

    return res.status(response.status).json(response.data);
  } catch (error) {
    console.error("❌ Error calculating time series:", error.message);

    // Check if error response has details
    const errorData = error.response?.data;
    const statusCode = error.response?.status || 500;

    return res.status(statusCode).json({
      error: errorData?.error || "Failed to calculate time series",
      details: errorData?.details || error.message,
      suggestion: errorData?.suggestion,
      debugging: errorData?.debugging
    });
  }
});

module.exports = router;
