// Dashboard JavaScript Module
const Dashboard = {
    // Configuration
    config: {
        refreshInterval: 30000, // 30 seconds
        chartColors: {
            primary: '#007bff',
            success: '#28a745',
            warning: '#ffc107',
            danger: '#dc3545',
            info: '#17a2b8'
        }
    },
    
    // State
    state: {
        charts: {},
        refreshTimer: null,
        isUpdating: false
    },
    
    // Initialize dashboard
    init() {
        console.log('Initializing Dashboard...');
        this.setupEventListeners();
        this.initializeCharts();
        this.loadInitialData();
        this.startAutoRefresh();
    },
    
    // Setup event listeners
    setupEventListeners() {
        // Period selection for volume chart
        document.querySelectorAll('input[name="volume-period"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.updateVolumeChart(e.target.value);
            });
        });
        
        // Refresh buttons
        document.getElementById('refresh-predictions-btn')?.addEventListener('click', () => {
            this.loadRecentPredictions();
        });
        
        document.getElementById('refresh-alerts-btn')?.addEventListener('click', () => {
            this.loadActiveAlerts();
        });
        
        document.getElementById('check-drift-btn')?.addEventListener('click', () => {
            this.checkDataDrift();
        });
    },
    
    // Initialize all charts
    initializeCharts() {
        this.initPredictionVolumeChart();
        this.initPerformanceChart();
        this.initResponseTimeChart();
        this.initDriftChart();
    },
    
    // Initialize prediction volume chart
    initPredictionVolumeChart() {
        const ctx = document.getElementById('predictionVolumeChart');
        if (!ctx) return;
        
        this.state.charts.volume = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Predictions per Hour',
                    data: [],
                    borderColor: this.config.chartColors.primary,
                    backgroundColor: this.config.chartColors.primary + '20',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    },
    
    // Initialize performance chart
    initPerformanceChart() {
        const ctx = document.getElementById('performanceChart');
        if (!ctx) return;
        
        this.state.charts.performance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Accuracy', 'Remaining'],
                datasets: [{
                    data: [0, 100],
                    backgroundColor: [
                        this.config.chartColors.success,
                        '#e9ecef'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    },
    
    // Initialize response time chart
    initResponseTimeChart() {
        const ctx = document.getElementById('responseTimeChart');
        if (!ctx) return;
        
        this.state.charts.responseTime = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Response Time (ms)',
                    data: [],
                    backgroundColor: this.config.chartColors.info + '80',
                    borderColor: this.config.chartColors.info,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Response Time (ms)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    },
    
    // Initialize drift chart
    initDriftChart() {
        const ctx = document.getElementById('driftChart');
        if (!ctx) return;
        
        this.state.charts.drift = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Drift Score',
                    data: [],
                    borderColor: this.config.chartColors.warning,
                    backgroundColor: this.config.chartColors.warning + '20',
                    pointBackgroundColor: this.config.chartColors.warning,
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: this.config.chartColors.warning
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            stepSize: 0.2
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    },
    
    // Load initial dashboard data
    async loadInitialData() {
        this.state.isUpdating = true;
        this.updateLastRefreshTime();
        
        try {
            await Promise.all([
                this.loadDashboardMetrics(),
                this.loadRecentPredictions(),
                this.loadActiveAlerts(),
                this.checkDataDrift()
            ]);
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showError('Failed to load dashboard data');
        } finally {
            this.state.isUpdating = false;
        }
    },
    
    // Load dashboard metrics
    async loadDashboardMetrics() {
        try {
            const response = await fetch('/metrics');
            const data = await response.json();
            
            if (response.ok) {
                this.updateStatusCards(data);
                this.updateCharts(data);
            } else {
                throw new Error(data.error || 'Failed to load metrics');
            }
        } catch (error) {
            console.error('Error loading metrics:', error);
            throw error;
        }
    },
    
    // Update status cards
    updateStatusCards(data) {
        // Active models count
        const activeModelsEl = document.getElementById('active-models-count');
        if (activeModelsEl) {
            activeModelsEl.textContent = data.model_count || 0;
        }
        
        // Predictions today
        const predictionsTodayEl = document.getElementById('predictions-today');
        if (predictionsTodayEl) {
            predictionsTodayEl.textContent = data.predictions_today || 0;
        }
        
        // Average response time
        const avgResponseTimeEl = document.getElementById('avg-response-time');
        if (avgResponseTimeEl) {
            const avgTime = data.avg_response_time || 0;
            avgResponseTimeEl.textContent = `${avgTime}ms`;
        }
        
        // Active alerts count
        const activeAlertsEl = document.getElementById('active-alerts-count');
        if (activeAlertsEl) {
            activeAlertsEl.textContent = data.active_alerts || 0;
        }
    },
    
    // Update charts with new data
    updateCharts(data) {
        // Update volume chart
        if (this.state.charts.volume && data.prediction_volume) {
            const chart = this.state.charts.volume;
            chart.data.labels = data.prediction_volume.labels || [];
            chart.data.datasets[0].data = data.prediction_volume.data || [];
            chart.update('none');
        }
        
        // Update performance chart
        if (this.state.charts.performance && data.current_accuracy !== undefined) {
            const chart = this.state.charts.performance;
            const accuracy = Math.round(data.current_accuracy * 100);
            chart.data.datasets[0].data = [accuracy, 100 - accuracy];
            chart.update('none');
        }
        
        // Update response time chart
        if (this.state.charts.responseTime && data.response_times) {
            const chart = this.state.charts.responseTime;
            chart.data.labels = data.response_times.labels || [];
            chart.data.datasets[0].data = data.response_times.data || [];
            chart.update('none');
        }
    },
    
    // Update volume chart for specific period
    async updateVolumeChart(hours) {
        try {
            const response = await fetch(`/metrics?hours=${hours}`);
            const data = await response.json();
            
            if (response.ok && this.state.charts.volume && data.prediction_volume) {
                const chart = this.state.charts.volume;
                chart.data.labels = data.prediction_volume.labels || [];
                chart.data.datasets[0].data = data.prediction_volume.data || [];
                chart.update();
            }
        } catch (error) {
            console.error('Error updating volume chart:', error);
        }
    },
    
    // Load recent predictions
    async loadRecentPredictions() {
        try {
            const response = await fetch('/metrics');
            const data = await response.json();
            
            if (response.ok) {
                this.updatePredictionsTable(data.recent_predictions || []);
            } else {
                throw new Error(data.error || 'Failed to load predictions');
            }
        } catch (error) {
            console.error('Error loading predictions:', error);
            this.updatePredictionsTable([]);
        }
    },
    
    // Update predictions table
    updatePredictionsTable(predictions) {
        const tableBody = document.getElementById('recent-predictions-table');
        if (!tableBody) return;
        
        if (predictions.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No recent predictions</td></tr>';
            return;
        }
        
        const rows = predictions.slice(0, 10).map(pred => {
            const time = new Date(pred.timestamp).toLocaleTimeString();
            const confidence = pred.confidence ? `${Math.round(pred.confidence * 100)}%` : 'N/A';
            const prediction = typeof pred.prediction === 'object' ? 
                JSON.stringify(pred.prediction) : pred.prediction;
            
            return `
                <tr>
                    <td>${time}</td>
                    <td><span class="badge bg-secondary">${pred.model_id}</span></td>
                    <td>${prediction}</td>
                    <td>${confidence}</td>
                    <td>${pred.response_time_ms}ms</td>
                </tr>
            `;
        }).join('');
        
        tableBody.innerHTML = rows;
    },
    
    // Load active alerts
    async loadActiveAlerts() {
        try {
            const response = await fetch('/alerts');
            const data = await response.json();
            
            if (response.ok) {
                this.updateAlertsDisplay(data.alerts || []);
            } else {
                throw new Error(data.error || 'Failed to load alerts');
            }
        } catch (error) {
            console.error('Error loading alerts:', error);
            this.updateAlertsDisplay([]);
        }
    },
    
    // Update alerts display
    updateAlertsDisplay(alerts) {
        const alertsList = document.getElementById('alerts-list');
        if (!alertsList) return;
        
        if (alerts.length === 0) {
            alertsList.innerHTML = '<div class="text-center text-muted">No active alerts</div>';
            return;
        }
        
        const alertsHtml = alerts.map(alert => {
            const time = new Date(alert.timestamp).toLocaleTimeString();
            const severityClass = alert.severity === 'high' ? 'danger' : 
                                 alert.severity === 'medium' ? 'warning' : 'info';
            
            return `
                <div class="alert-item ${severityClass}">
                    <h6>${alert.title}</h6>
                    <p>${alert.message}</p>
                    <small class="text-muted">${time}</small>
                    <button class="btn btn-sm btn-outline-secondary float-end" 
                            onclick="Dashboard.resolveAlert('${alert.id}')">
                        Resolve
                    </button>
                </div>
            `;
        }).join('');
        
        alertsList.innerHTML = alertsHtml;
    },
    
    // Check data drift
    async checkDataDrift() {
        const driftStatus = document.getElementById('drift-status');
        const driftChart = document.getElementById('driftChart');
        
        if (!driftStatus) return;
        
        // Show loading state
        driftStatus.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Checking drift...</span>
                </div>
                <p class="mt-2">Checking for data drift...</p>
            </div>
        `;
        driftStatus.classList.remove('d-none');
        driftChart?.classList.add('d-none');
        
        try {
            // Get current model first
            const statusResponse = await fetch('/model/status');
            const statusData = await statusResponse.json();
            
            if (!statusData.current_model) {
                driftStatus.innerHTML = `
                    <div class="drift-status insufficient-data">
                        <i class="fas fa-info-circle"></i>
                        <p class="mb-0">No active model for drift detection</p>
                    </div>
                `;
                return;
            }
            
            const modelId = statusData.current_model.model_id;
            const response = await fetch(`/metrics/drift/${modelId}?hours=1`);
            const data = await response.json();
            
            if (response.ok) {
                this.updateDriftDisplay(data);
            } else {
                throw new Error(data.message || 'Failed to check drift');
            }
        } catch (error) {
            console.error('Error checking drift:', error);
            driftStatus.innerHTML = `
                <div class="drift-status insufficient-data">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p class="mb-0">Error checking drift: ${error.message}</p>
                </div>
            `;
        }
    },
    
    // Update drift display
    updateDriftDisplay(driftData) {
        const driftStatus = document.getElementById('drift-status');
        const driftChart = document.getElementById('driftChart');
        
        if (!driftStatus) return;
        
        if (driftData.message) {
            // Insufficient data case
            driftStatus.innerHTML = `
                <div class="drift-status insufficient-data">
                    <i class="fas fa-info-circle"></i>
                    <p class="mb-0">${driftData.message}</p>
                </div>
            `;
            driftChart?.classList.add('d-none');
            return;
        }
        
        // Show drift status
        const hasDrift = driftData.has_drift;
        const driftScore = driftData.drift_score || 0;
        const statusClass = hasDrift ? 'drift-detected' : 'no-drift';
        const icon = hasDrift ? 'fa-exclamation-triangle' : 'fa-check-circle';
        const message = hasDrift ? 
            `Data drift detected (score: ${driftScore.toFixed(3)})` :
            `No drift detected (score: ${driftScore.toFixed(3)})`;
        
        driftStatus.innerHTML = `
            <div class="drift-status ${statusClass}">
                <i class="fas ${icon}"></i>
                <p class="mb-0">${message}</p>
            </div>
        `;
        
        // Update drift chart if feature drifts are available
        if (driftData.feature_drifts && this.state.charts.drift) {
            const features = Object.keys(driftData.feature_drifts);
            const scores = Object.values(driftData.feature_drifts);
            
            const chart = this.state.charts.drift;
            chart.data.labels = features;
            chart.data.datasets[0].data = scores;
            chart.update();
            
            driftChart?.classList.remove('d-none');
        }
    },
    
    // Resolve alert
    async resolveAlert(alertId) {
        try {
            const response = await fetch(`/alerts/${alertId}/resolve`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.loadActiveAlerts(); // Refresh alerts
            } else {
                const data = await response.json();
                throw new Error(data.error || 'Failed to resolve alert');
            }
        } catch (error) {
            console.error('Error resolving alert:', error);
            this.showError('Failed to resolve alert');
        }
    },
    
    // Start auto-refresh
    startAutoRefresh() {
        this.state.refreshTimer = setInterval(() => {
            if (!this.state.isUpdating) {
                this.loadInitialData();
            }
        }, this.config.refreshInterval);
    },
    
    // Stop auto-refresh
    stopAutoRefresh() {
        if (this.state.refreshTimer) {
            clearInterval(this.state.refreshTimer);
            this.state.refreshTimer = null;
        }
    },
    
    // Update last refresh time
    updateLastRefreshTime() {
        const updateTimeEl = document.getElementById('update-time');
        if (updateTimeEl) {
            updateTimeEl.textContent = new Date().toLocaleTimeString();
        }
    },
    
    // Show error message
    showError(message) {
        const alertBanner = document.getElementById('alert-banner');
        const alertMessage = document.getElementById('alert-message');
        
        if (alertBanner && alertMessage) {
            alertMessage.textContent = message;
            alertBanner.classList.remove('d-none');
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                alertBanner.classList.add('d-none');
            }, 5000);
        }
    },
    
    // Cleanup when page unloads
    cleanup() {
        this.stopAutoRefresh();
        
        // Destroy charts
        Object.values(this.state.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        
        this.state.charts = {};
    }
};

// Models page functionality
const ModelsPage = {
    init() {
        console.log('Initializing Models Page...');
        this.setupEventListeners();
        this.loadModels();
        this.loadCurrentModel();
    },
    
    setupEventListeners() {
        // Refresh models
        document.getElementById('refresh-models-btn')?.addEventListener('click', () => {
            this.loadModels();
        });
        
        // Manual training form
        document.getElementById('manual-training-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.startTraining();
        });
        
        // Load sample data for training
        document.getElementById('load-sample-data-btn')?.addEventListener('click', () => {
            this.loadSampleTrainingData();
        });
    },
    
    async loadModels() {
        try {
            const response = await fetch('/model/status');
            const data = await response.json();
            
            if (response.ok) {
                this.updateModelsTable(data.available_models || []);
                this.updateModelsCount(data.available_models?.length || 0);
            } else {
                throw new Error(data.error || 'Failed to load models');
            }
        } catch (error) {
            console.error('Error loading models:', error);
            this.updateModelsTable([]);
        }
    },
    
    async loadCurrentModel() {
        try {
            const response = await fetch('/model/status');
            const data = await response.json();
            
            if (response.ok) {
                this.updateCurrentModelInfo(data.current_model);
            }
        } catch (error) {
            console.error('Error loading current model:', error);
        }
    },
    
    updateModelsTable(models) {
        const tableBody = document.getElementById('models-table');
        if (!tableBody) return;
        
        if (models.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No models available</td></tr>';
            return;
        }
        
        const rows = models.map(model => {
            const createdAt = new Date(model.created_at).toLocaleString();
            const accuracy = model.validation_accuracy ? 
                `${Math.round(model.validation_accuracy * 100)}%` : 'N/A';
            const features = model.feature_names?.length || 0;
            
            return `
                <tr>
                    <td><code>${model.model_id}</code></td>
                    <td><span class="badge bg-info">${model.model_type}</span></td>
                    <td>${createdAt}</td>
                    <td>${accuracy}</td>
                    <td>${features} features</td>
                    <td><span class="model-status"><span class="model-status-dot active"></span> Active</span></td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary" onclick="ModelsPage.viewModelDetails('${model.model_id}')">
                            View
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
        
        tableBody.innerHTML = rows;
    },
    
    updateCurrentModelInfo(currentModel) {
        const infoContainer = document.getElementById('current-model-info');
        if (!infoContainer) return;
        
        if (!currentModel) {
            infoContainer.innerHTML = '<div class="text-center text-muted">No active model</div>';
            return;
        }
        
        const createdAt = new Date(currentModel.created_at).toLocaleString();
        const accuracy = currentModel.validation_accuracy ? 
            `${Math.round(currentModel.validation_accuracy * 100)}%` : 'N/A';
        
        infoContainer.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <strong>Model ID:</strong><br>
                    <code>${currentModel.model_id}</code>
                </div>
                <div class="col-md-2">
                    <strong>Type:</strong><br>
                    <span class="badge bg-info">${currentModel.model_type}</span>
                </div>
                <div class="col-md-2">
                    <strong>Accuracy:</strong><br>
                    ${accuracy}
                </div>
                <div class="col-md-3">
                    <strong>Created:</strong><br>
                    ${createdAt}
                </div>
                <div class="col-md-2">
                    <strong>Features:</strong><br>
                    ${currentModel.feature_names?.length || 0}
                </div>
            </div>
        `;
    },
    
    updateModelsCount(count) {
        const countEl = document.getElementById('models-count');
        if (countEl) {
            countEl.textContent = `${count} model${count !== 1 ? 's' : ''}`;
        }
    },
    
    startTraining() {
        // This would trigger model training
        // For now, show a placeholder modal
        const modal = new bootstrap.Modal(document.getElementById('trainingModal'));
        modal.show();
        
        // Simulate training progress
        setTimeout(() => {
            modal.hide();
            this.loadModels(); // Refresh models list
        }, 3000);
    },
    
    viewModelDetails(modelId) {
        // Show model details in modal
        const modal = new bootstrap.Modal(document.getElementById('modelDetailsModal'));
        const content = document.getElementById('model-details-content');
        
        if (content) {
            content.innerHTML = `
                <div class="text-center">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Loading model details...</p>
                </div>
            `;
        }
        
        modal.show();
        
        // Load model details (placeholder)
        setTimeout(() => {
            if (content) {
                content.innerHTML = `
                    <h6>Model: ${modelId}</h6>
                    <p>Detailed model information would be displayed here.</p>
                `;
            }
        }, 1000);
    }
};

// API Test page functionality
const ApiTestPage = {
    init() {
        console.log('Initializing API Test Page...');
        this.setupEventListeners();
        this.loadAvailableModels();
        this.performInitialHealthCheck();
    },
    
    setupEventListeners() {
        // Health check button
        document.getElementById('health-check-btn')?.addEventListener('click', () => {
            this.performHealthCheck();
        });
        
        // Test endpoint buttons
        document.querySelectorAll('.test-endpoint-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const endpoint = e.target.dataset.endpoint;
                this.testEndpoint(endpoint);
            });
        });
        
        // Prediction test form
        document.getElementById('prediction-test-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.testPrediction();
        });
        
        // Load sample data buttons
        document.querySelectorAll('.load-sample-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const sampleType = e.target.dataset.sample;
                this.loadSampleData(sampleType);
            });
        });
        
        // Clear input
        document.getElementById('clear-input-btn')?.addEventListener('click', () => {
            document.getElementById('features-input').value = '';
            document.getElementById('prediction-response').innerHTML = `
                <div class="text-muted text-center">
                    <i class="fas fa-arrow-left"></i>
                    <p class="mt-2">Submit a prediction request to see the response</p>
                </div>
            `;
        });
        
        // Batch testing
        document.getElementById('run-batch-test-btn')?.addEventListener('click', () => {
            this.runBatchTest();
        });
    },
    
    // Perform initial health check
    async performInitialHealthCheck() {
        await this.performHealthCheck();
    },
    
    // Perform health check
    async performHealthCheck() {
        const banner = document.getElementById('health-status-banner');
        const statusText = document.getElementById('health-status-text');
        const checkTime = document.getElementById('health-check-time');
        
        if (!banner || !statusText) return;
        
        // Show banner and loading state
        banner.className = 'alert alert-info';
        banner.classList.remove('d-none');
        statusText.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Checking API health...';
        
        try {
            const startTime = Date.now();
            const response = await fetch('/health');
            const responseTime = Date.now() - startTime;
            const data = await response.json();
            
            if (response.ok) {
                banner.className = 'alert alert-success';
                statusText.innerHTML = `<i class="fas fa-check-circle me-2"></i>API is healthy (${responseTime}ms)`;
                if (checkTime) {
                    checkTime.textContent = `Last checked: ${new Date().toLocaleTimeString()}`;
                }
            } else {
                throw new Error(data.error || 'Health check failed');
            }
        } catch (error) {
            banner.className = 'alert alert-danger';
            statusText.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>API health check failed: ${error.message}`;
            if (checkTime) {
                checkTime.textContent = `Last checked: ${new Date().toLocaleTimeString()}`;
            }
        }
    },
    
    // Test specific endpoint
    async testEndpoint(endpoint) {
        const resultsDiv = document.getElementById('quick-test-results');
        if (!resultsDiv) return;
        
        // Show loading state
        resultsDiv.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Testing ${endpoint}...</span>
                </div>
                <p class="mt-2">Testing ${endpoint} endpoint...</p>
            </div>
        `;
        
        try {
            let response, data, requestBody = null;
            const startTime = Date.now();
            
            switch (endpoint) {
                case 'health':
                    response = await fetch('/health');
                    break;
                    
                case 'health-detailed':
                    response = await fetch('/health/detailed');
                    break;
                    
                case 'health-readiness':
                    response = await fetch('/health/readiness');
                    break;
                    
                case 'health-liveness':
                    response = await fetch('/health/liveness');
                    break;
                    
                case 'model-status':
                    response = await fetch('/model/status');
                    break;
                    
                case 'metrics':
                    response = await fetch('/metrics?hours=1');
                    break;
                    
                case 'alerts':
                    response = await fetch('/alerts');
                    break;
                    
                case 'predict':
                    requestBody = {
                        features: {
                            feature_1: 1.5,
                            feature_2: 2.0,
                            feature_3: 1.2,
                            feature_4: 0.8
                        }
                    };
                    response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(requestBody)
                    });
                    break;
                    
                case 'retrain':
                    // Get current model first
                    const statusResponse = await fetch('/model/status');
                    const statusData = await statusResponse.json();
                    const modelId = statusData.current_model?.model_id || 'demo_model';
                    
                    requestBody = {
                        model_id: modelId,
                        notes: 'Test retraining from API documentation'
                    };
                    response = await fetch('/model/retrain', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(requestBody)
                    });
                    break;
                    
                default:
                    throw new Error(`Unknown endpoint: ${endpoint}`);
            }
            
            const responseTime = Date.now() - startTime;
            data = await response.json();
            
            // Display results
            this.displayTestResults(endpoint, response.status, data, responseTime, requestBody);
            
        } catch (error) {
            resultsDiv.innerHTML = `
                <div class="alert alert-danger">
                    <h6><i class="fas fa-exclamation-triangle"></i> Test Failed</h6>
                    <p class="mb-0">Error testing ${endpoint}: ${error.message}</p>
                </div>
            `;
        }
    },
    
    // Display test results
    displayTestResults(endpoint, status, data, responseTime, requestBody = null) {
        const resultsDiv = document.getElementById('quick-test-results');
        if (!resultsDiv) return;
        
        const statusClass = status >= 200 && status < 300 ? 'success' : 
                           status >= 400 && status < 500 ? 'warning' : 'danger';
        const statusIcon = status >= 200 && status < 300 ? 'check-circle' : 
                          status >= 400 && status < 500 ? 'exclamation-triangle' : 'times-circle';
        
        let html = `
            <div class="alert alert-${statusClass}">
                <div class="d-flex justify-content-between align-items-start mb-2">
                    <h6><i class="fas fa-${statusIcon}"></i> ${endpoint.toUpperCase()} Test Result</h6>
                    <div class="text-end">
                        <span class="badge bg-secondary">HTTP ${status}</span>
                        <span class="badge bg-info">${responseTime}ms</span>
                    </div>
                </div>
        `;
        
        if (requestBody) {
            html += `
                <div class="mb-2">
                    <strong>Request Body:</strong>
                    <pre class="bg-light p-2 rounded mt-1"><code>${JSON.stringify(requestBody, null, 2)}</code></pre>
                </div>
            `;
        }
        
        html += `
                <div>
                    <strong>Response:</strong>
                    <pre class="bg-light p-2 rounded mt-1"><code>${JSON.stringify(data, null, 2)}</code></pre>
                </div>
            </div>
        `;
        
        resultsDiv.innerHTML = html;
    },
    
    async loadAvailableModels() {
        try {
            const response = await fetch('/model/status');
            const data = await response.json();
            
            if (response.ok) {
                this.updateModelSelect(data.available_models || []);
            }
        } catch (error) {
            console.error('Error loading models:', error);
        }
    },
    
    updateModelSelect(models) {
        const select = document.getElementById('model-id-input');
        if (!select) return;
        
        // Clear existing options except the first one
        select.innerHTML = '<option value="">Use latest model</option>';
        
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.model_id;
            option.textContent = `${model.model_id} (${model.model_type})`;
            select.appendChild(option);
        });
    },
    
    async testPrediction() {
        const modelId = document.getElementById('model-id-input').value;
        const featuresText = document.getElementById('features-input').value;
        const responseDiv = document.getElementById('prediction-response');
        
        if (!featuresText.trim()) {
            this.showResponse({
                error: 'Please enter feature values'
            }, false);
            return;
        }
        
        // Show loading state
        responseDiv.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Making prediction...</span>
                </div>
                <p class="mt-2">Making prediction...</p>
            </div>
        `;
        
        try {
            const features = JSON.parse(featuresText);
            const payload = { features };
            if (modelId) payload.model_id = modelId;
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();
            this.showResponse(data, response.ok);
            
        } catch (error) {
            this.showResponse({
                error: 'Invalid JSON format',
                details: error.message
            }, false);
        }
    },
    
    showResponse(data, isSuccess) {
        const responseDiv = document.getElementById('prediction-response');
        if (!responseDiv) return;
        
        const statusClass = isSuccess ? 'success' : 'danger';
        const statusIcon = isSuccess ? 'check-circle' : 'exclamation-triangle';
        
        let content = `
            <div class="alert alert-${statusClass}">
                <i class="fas fa-${statusIcon}"></i>
                <strong>${isSuccess ? 'Success' : 'Error'}</strong>
            </div>
        `;
        
        if (isSuccess) {
            content += `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Prediction Result</h6>
                        <div class="bg-light p-3 rounded">
                            <strong>${JSON.stringify(data.prediction)}</strong>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h6>Details</h6>
                        <ul class="list-unstyled">
                            <li><strong>Model:</strong> ${data.model_id}</li>
                            <li><strong>Type:</strong> ${data.model_type}</li>
                            <li><strong>Confidence:</strong> ${data.confidence ? Math.round(data.confidence * 100) + '%' : 'N/A'}</li>
                            <li><strong>Response Time:</strong> ${data.response_time_ms}ms</li>
                        </ul>
                    </div>
                </div>
            `;
        } else {
            content += `
                <h6>Error Details</h6>
                <pre class="bg-light p-3 rounded">${JSON.stringify(data, null, 2)}</pre>
            `;
        }
        
        responseDiv.innerHTML = content;
    },
    
    loadSampleData(sampleType) {
        const featuresInput = document.getElementById('features-input');
        if (!featuresInput) return;
        
        const samples = {
            iris: {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            classification: {
                "feature_1": 1.5,
                "feature_2": -0.8,
                "feature_3": 2.1,
                "feature_4": 0.3
            },
            regression: {
                "x1": 10.5,
                "x2": 25.3,
                "x3": -5.7,
                "x4": 8.9
            }
        };
        
        const sample = samples[sampleType];
        if (sample) {
            featuresInput.value = JSON.stringify(sample, null, 2);
        }
    },
    
    async runBatchTest() {
        const batchSize = parseInt(document.getElementById('batch-size').value) || 10;
        const batchDelay = parseInt(document.getElementById('batch-delay').value) || 100;
        
        const progressDiv = document.getElementById('batch-progress');
        const resultsDiv = document.getElementById('batch-results');
        const progressBar = document.getElementById('batch-progress-bar');
        const statusSpan = document.getElementById('batch-status');
        
        // Show progress
        progressDiv?.classList.remove('d-none');
        resultsDiv?.classList.add('d-none');
        
        let successCount = 0;
        let errorCount = 0;
        let totalTime = 0;
        const startTime = Date.now();
        
        // Sample data for batch testing
        const sampleFeatures = {
            "feature_1": 1.0,
            "feature_2": 2.0,
            "feature_3": 3.0,
            "feature_4": 4.0
        };
        
        for (let i = 0; i < batchSize; i++) {
            try {
                const requestStart = Date.now();
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ features: sampleFeatures })
                });
                
                const requestTime = Date.now() - requestStart;
                totalTime += requestTime;
                
                if (response.ok) {
                    successCount++;
                } else {
                    errorCount++;
                }
                
                // Update progress
                const progress = ((i + 1) / batchSize) * 100;
                if (progressBar) progressBar.style.width = `${progress}%`;
                if (statusSpan) statusSpan.textContent = `${i + 1}/${batchSize} requests completed`;
                
                // Delay between requests
                if (i < batchSize - 1 && batchDelay > 0) {
                    await new Promise(resolve => setTimeout(resolve, batchDelay));
                }
                
            } catch (error) {
                errorCount++;
            }
        }
        
        const endTime = Date.now();
        const totalTestTime = endTime - startTime;
        const avgResponseTime = successCount > 0 ? Math.round(totalTime / successCount) : 0;
        
        // Show results
        progressDiv?.classList.add('d-none');
        resultsDiv?.classList.remove('d-none');
        
        document.getElementById('batch-success-count').textContent = successCount;
        document.getElementById('batch-error-count').textContent = errorCount;
        document.getElementById('batch-avg-time').textContent = `${avgResponseTime}ms`;
        document.getElementById('batch-total-time').textContent = `${Math.round(totalTestTime / 1000)}s`;
    }
};

// Cleanup when page unloads
window.addEventListener('beforeunload', () => {
    if (typeof Dashboard !== 'undefined') {
        Dashboard.cleanup();
    }
});

// Export for global access
window.Dashboard = Dashboard;
window.ModelsPage = ModelsPage;
window.ApiTestPage = ApiTestPage;