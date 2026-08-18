const getHealthStatus = (req, res) => {
  res.json({
    status: "OK",
    message: "Bio-Block JavaScript Backend API is running",
    timestamp: new Date().toISOString(),
    version: "1.0.0",
    endpoints: ["/api/health", "/api/anonymize"],
  });
};

module.exports = {
  getHealthStatus,
};
