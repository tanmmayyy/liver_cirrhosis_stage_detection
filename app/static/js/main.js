// Enhanced Frontend JavaScript for Liver Cirrhosis Detection
// This file should be saved as app/static/js/main.js

class LiverCirrhosisApp {
    constructor() {
        this.apiBase = '/api';
        this.form = document.getElementById('predictionForm');
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.predictionResult = document.getElementById('predictionResult');
        this.stageInfo = document.getElementById('stageInfo');
        this.featureImportance = document.getElementById('featureImportance');
        this.errorMessage = document.getElementById('errorMessage');
        this.successMessage = document.getElementById('successMessage');
        
        this.initializeEventListeners();
        this.loadFormData();
        this.checkModelStatus();
    }

    initializeEventListeners() {
        // Form submission
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        
        // Real-time validation
        const inputs = this.form.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('input', () => this.validateField(input));
            input.addEventListener('blur', () => this.validateField(input));
        });

        // Save form data on change
        inputs.forEach(input => {
            input.addEventListener('change', () => this.saveFormData());
        });

        // Add interactive effects
        this.addInteractiveEffects();
    }

    addInteractiveEffects() {
        // Form field hover effects
        document.querySelectorAll('.form-group').forEach(group => {
            const input = group.querySelector('input, select');
            
            group.addEventListener('mouseenter', () => {
                if (!input.matches(':focus')) {
                    group.style.transform = 'translateY(-2px)';
                    group.style.transition = 'transform 0.2s ease';
                }
            });

            group.addEventListener('mouseleave', () => {
                if (!input.matches(':focus')) {
                    group.style.transform = 'translateY(0)';
                }
            });
        });

        // Smooth scrolling for results
        document.addEventListener('scroll', this.handleScroll);
    }

    validateField(field) {
        const value = field.value;
        const name = field.name;
        
        // Remove previous validation classes
        field.classList.remove('valid', 'invalid', 'warning');
        
        if (!value) {
            field.style.borderColor = '#e2e8f0';
            return;
        }

        // Specific medical validations
        let isValid = true;
        let isWarning = false;
        
        switch(name) {
            case 'age':
                if (value < 18 || value > 100) {
                    isValid = false;
                } else if (value > 75) {
                    isWarning = true;
                }
                break;
                
            case 'bilirubin':
                if (value < 0.1 || value > 50) {
                    isValid = false;
                } else if (value > 3.0) {
                    isWarning = true;
                    this.showTooltip(field, 'Elevated bilirubin level detected');
                }
                break;
                
            case 'albumin':
                if (value < 1 || value > 6) {
                    isValid = false;
                } else if (value < 3.0) {
                    isWarning = true;
                    this.showTooltip(field, 'Low albumin level may indicate liver dysfunction');
                }
                break;
                
            case 'platelets':
                if (value < 50 || value > 600) {
                    isValid = false;
                } else if (value < 150) {
                    isWarning = true;
                    this.showTooltip(field, 'Low platelet count detected');
                }
                break;
                
            case 'prothrombin':
                if (value < 8 || value > 25) {
                    isValid = false;
                } else if (value > 13) {
                    isWarning = true;
                    this.showTooltip(field, 'Prolonged prothrombin time');
                }
                break;
        }

        // Apply validation styling
        if (isValid) {
            if (isWarning) {
                field.style.borderColor = '#f6ad55';
                field.classList.add('warning');
            } else {
                field.style.borderColor = '#48bb78';
                field.classList.add('valid');
            }
        } else {
            field.style.borderColor = '#f56565';
            field.classList.add('invalid');
        }
    }

    showTooltip(element, message) {
        // Create and show tooltip
        const tooltip = document.createElement('div');
        tooltip.className = 'validation-tooltip';
        tooltip.textContent = message;
        
        const rect = element.getBoundingClientRect();
        tooltip.style.cssText = `
            position: fixed;
            top: ${rect.top - 35}px;
            left: ${rect.left}px;
            background: #2d3748;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            z-index: 1000;
            animation: fadeIn 0.3s ease;
        `;
        
        document.body.appendChild(tooltip);
        
        setTimeout(() => {
            if (tooltip.parentNode) {
                tooltip.remove();
            }
        }, 3000);
    }

    async handleSubmit(e) {
        e.preventDefault();
        
        try {
            this.showLoading(true);
            this.hideMessages();
            
            const formData = new FormData(this.form);
            const patientData = Object.fromEntries(formData.entries());
            
            // Convert numeric fields
            const numericFields = ['age', 'n_days', 'bilirubin', 'cholesterol', 'albumin', 
                                 'copper', 'alk_phos', 'sgot', 'tryglicerides', 'platelets', 'prothrombin'];
            
            numericFields.forEach(field => {
                if (patientData[field]) {
                    patientData[field] = parseFloat(patientData[field]);
                }
            });

            // Make API call
            const response = await fetch(`${this.apiBase}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(patientData)
            });

            const result = await response.json();

            if (response.ok && result.success) {
                this.displayResults(result.prediction);
                this.showSuccess('Prediction completed successfully!');
                this.saveFormData(); // Save successful form data
            } else {
                throw new Error(result.error || 'Prediction failed');
            }
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showError(`Prediction failed: ${error.message}`);
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(prediction) {
        const { stage, confidence, probabilities, feature_importance } = prediction;
        
        // Update main result with animation
        this.predictionResult.className = `prediction-result stage-${stage}`;
        this.predictionResult.style.transform = 'scale(0.9)';
        this.predictionResult.style.opacity = '0';
        
        setTimeout(() => {
            this.predictionResult.innerHTML = `
                <div class="stage-number">${stage}</div>
                <div class="stage-label">Stage ${stage} Cirrhosis</div>
                <div class="confidence-score">Confidence: ${(confidence * 100).toFixed(1)}%</div>
            `;
            
            this.predictionResult.style.transform = 'scale(1)';
            this.predictionResult.style.opacity = '1';
            this.predictionResult.style.transition = 'all 0.3s ease';
        }, 100);

        // Update stage information
        const stageDescriptions = {
            1: {
                title: "Stage 1 - Mild Fibrosis",
                description: "Early stage with portal fibrosis. Liver function is generally preserved.",
                recommendations: [
                    "Regular monitoring every 6 months",
                    "Lifestyle modifications (diet, exercise)",
                    "Treat underlying causes",
                    "Avoid hepatotoxic medications"
                ],
                prognosis: "Good prognosis with proper management"
            },
            2: {
                title: "Stage 2 - Moderate Fibrosis", 
                description: "Progressive fibrosis with portal-to-portal bridging. Some liver function compromise.",
                recommendations: [
                    "Close monitoring every 3-4 months",
                    "Screen for esophageal varices",
                    "Consider treatment intensification",
                    "Monitor for complications"
                ],
                prognosis: "Requires careful monitoring and management"
            },
            3: {
                title: "Stage 3 - Severe Fibrosis/Cirrhosis",
                description: "Advanced fibrosis approaching or at cirrhosis. High risk of complications.",
                recommendations: [
                    "Specialist hepatology care required",
                    "Regular screening for complications",
                    "Consider liver transplant evaluation",
                    "Portal hypertension management"
                ],
                prognosis: "Requires intensive management and monitoring"
            }
        };

        const stageData = stageDescriptions[stage];
        this.stageInfo.innerHTML = `
            <h3><i class="fas fa-info-circle"></i> ${stageData.title}</h3>
            <p><strong>Description:</strong> ${stageData.description}</p>
            <p><strong>Prognosis:</strong> ${stageData.prognosis}</p>
            <div class="recommendations">
                <h4>Clinical Recommendations:</h4>
                <ul>
                    ${stageData.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        `;

        // Update probability visualization
        this.updateProbabilityChart(probabilities);

        // Update feature importance
        if (feature_importance && Object.keys(feature_importance).length > 0) {
            this.featureImportance.style.display = 'block';
            const importanceList = document.getElementById('importanceList');
            importanceList.innerHTML = '';
            
            Object.entries(feature_importance)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 5) // Top 5 features
                .forEach(([feature, importance]) => {
                    const featureDiv = document.createElement('div');
                    featureDiv.className = 'feature-bar';
                    
                    // Clean feature names for display
                    const displayName = this.formatFeatureName(feature);
                    
                    featureDiv.innerHTML = `
                        <div class="feature-name">${displayName}</div>
                        <div class="feature-progress">
                            <div class="feature-fill" style="width: 0%; animation: fillBar 1s ease forwards ${Math.random() * 0.5}s"></div>
                        </div>
                        <div class="feature-value">${(importance * 100).toFixed(0)}%</div>
                    `;
                    importanceList.appendChild(featureDiv);
                    
                    // Animate the fill
                    setTimeout(() => {
                        const fill = featureDiv.querySelector('.feature-fill');
                        fill.style.width = `${importance * 100}%`;
                    }, 100);
                });
        }

        // Scroll to results
        this.scrollToResults();
    }

    updateProbabilityChart(probabilities) {
        // Create or update probability visualization
        let chartContainer = document.getElementById('probabilityChart');
        if (!chartContainer) {
            chartContainer = document.createElement('div');
            chartContainer.id = 'probabilityChart';
            chartContainer.className = 'probability-chart';
            this.predictionResult.appendChild(chartContainer);
        }

        chartContainer.innerHTML = `
            <h4 style="margin: 15px 0 10px 0; color: white;">Stage Probabilities</h4>
            <div class="prob-bars">
                ${Object.entries(probabilities).map(([stage, prob]) => `
                    <div class="prob-item">
                        <span class="prob-label">${stage.replace('stage_', 'Stage ')}</span>
                        <div class="prob-bar">
                            <div class="prob-fill" style="width: ${prob * 100}%"></div>
                        </div>
                        <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
                    </div>
                `).join('')}
            </div>
        `;
    }

    formatFeatureName(feature) {
        // Convert technical feature names to readable format
        const nameMap = {
            'Bilirubin': 'Bilirubin Level',
            'Albumin': 'Albumin Level',
            'Prothrombin': 'Prothrombin Time',
            'Copper': 'Copper Level',
            'SGOT': 'SGOT Enzyme',
            'Alk_Phos': 'Alkaline Phosphatase',
            'Platelets': 'Platelet Count',
            'Age_Years': 'Patient Age',
            'Bilirubin_Albumin_Ratio': 'Bilirubin/Albumin Ratio',
            'Liver_Function_Score': 'Liver Function Score',
            'Portal_Hypertension_Score': 'Portal Hypertension Score'
        };
        
        return nameMap[feature] || feature.replace(/_/g, ' ');
    }

    scrollToResults() {
        setTimeout(() => {
            this.predictionResult.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'nearest' 
            });
        }, 500);
    }

    async checkModelStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const status = await response.json();
            
            if (!status.model_ready) {
                this.showError('Model not available. Please contact administrator.');
            }
        } catch (error) {
            console.error('Health check failed:', error);
        }
    }

    saveFormData() {
        const formData = new FormData(this.form);
        const data = Object.fromEntries(formData.entries());
        localStorage.setItem('liverCirrhosisFormData', JSON.stringify(data));
    }

    loadFormData() {
        const savedData = localStorage.getItem('liverCirrhosisFormData');
        if (savedData) {
            try {
                const data = JSON.parse(savedData);
                Object.entries(data).forEach(([key, value]) => {
                    const field = this.form.querySelector(`[name="${key}"]`);
                    if (field) {
                        field.value = value;
                    }
                });
            } catch (error) {
                console.error('Error loading saved form data:', error);
            }
        }
    }

    showLoading(show) {
        this.loadingSpinner.style.display = show ? 'block' : 'none';
        
        const button = this.form.querySelector('.predict-btn');
        button.disabled = show;
        
        if (show) {
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            button.style.background = 'linear-gradient(135deg, #a0aec0, #718096)';
        } else {
            button.innerHTML = '<i class="fas fa-brain"></i> Predict Cirrhosis Stage';
            button.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
        }
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.style.display = 'block';
        this.errorMessage.style.animation = 'slideIn 0.3s ease';
        
        setTimeout(() => this.hideMessages(), 5000);
    }

    showSuccess(message) {
        this.successMessage.textContent = message;
        this.successMessage.style.display = 'block';
        this.successMessage.style.animation = 'slideIn 0.3s ease';
        
        setTimeout(() => this.hideMessages(), 3000);
    }

    hideMessages() {
        this.errorMessage.style.display = 'none';
        this.successMessage.style.display = 'none';
    }

    handleScroll() {
        // Add parallax effect to header
        const header = document.querySelector('.header');
        const scrolled = window.pageYOffset;
        const rate = scrolled * -0.5;
        
        if (header) {
            header.style.transform = `translateY(${rate}px)`;
        }
    }
}

// Additional CSS animations and styles
const additionalStyles = `
<style>
@keyframes fillBar {
    from { width: 0%; }
    to { width: var(--target-width); }
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.validation-tooltip {
    animation: fadeIn 0.3s ease;
}

.form-group input.valid,
.form-group select.valid {
    border-color: #48bb78 !important;
    box-shadow: 0 0 0 3px rgba(72, 187, 120, 0.1);
}

.form-group input.invalid,
.form-group select.invalid {
    border-color: #f56565 !important;
    box-shadow: 0 0 0 3px rgba(245, 101, 101, 0.1);
}

.form-group input.warning,
.form-group select.warning {
    border-color: #f6ad55 !important;
    box-shadow: 0 0 0 3px rgba(246, 173, 85, 0.1);
}

.probability-chart {
    margin-top: 20px;
    padding: 15px;
    background: rgba(0,0,0,0.1);
    border-radius: 10px;
}

.prob-bars {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.prob-item {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.9rem;
}

.prob-label {
    width: 60px;
    font-weight: 500;
}

.prob-bar {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.2);
    border-radius: 3px;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    background: rgba(255,255,255,0.8);
    border-radius: 3px;
    transition: width 1s ease;
}

.prob-value {
    width: 45px;
    text-align: right;
    font-weight: 600;
}

.recommendations ul {
    margin: 10px 0;
    padding-left: 20px;
}

.recommendations li {
    margin: 5px 0;
    line-height: 1.4;
}

.feature-fill {
    transition: width 1s ease-out;
}

.loading-pulse {
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
</style>
`;

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Add additional styles
    document.head.insertAdjacentHTML('beforeend', additionalStyles);
    
    // Initialize the app
    new LiverCirrhosisApp();
    
    // Add some interactive animations
    document.querySelectorAll('.form-group input, .form-group select').forEach(element => {
        element.addEventListener('focus', function() {
            this.parentElement.style.transform = 'translateY(-2px)';
            this.parentElement.style.transition = 'transform 0.2s ease';
        });
        
        element.addEventListener('blur', function() {
            this.parentElement.style.transform = 'translateY(0)';
        });
    });
});

// Export for potential use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LiverCirrhosisApp;
}