/**
 * Product Classifier - Main JavaScript
 */

// Utility functions
const Utils = {
    // Format bytes to human readable format
    formatBytes: function(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    },
    
    // Debounce function for performance
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Show toast notification
    showToast: function(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-bg-${type} border-0 position-fixed top-0 end-0 m-3`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        document.body.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }
};

// API Client
const APIClient = {
    // Classify image
    classify: async function(formData) {
        const response = await fetch('/classify', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Classification failed');
        }
        
        return await response.json();
    },
    
    // Get categories
    getCategories: async function() {
        const response = await fetch('/api/categories');
        
        if (!response.ok) {
            throw new Error('Failed to fetch categories');
        }
        
        return await response.json();
    },
    
    // Health check
    healthCheck: async function() {
        const response = await fetch('/api/health');
        
        if (!response.ok) {
            throw new Error('Service is unhealthy');
        }
        
        return await response.json();
    },
    
    // Clear cache
    clearCache: async function() {
        const response = await fetch('/api/cache/clear', {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Failed to clear cache');
        }
        
        return await response.json();
    }
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Product Classifier loaded');
    
    // Check service health on load
    APIClient.healthCheck()
        .then(data => {
            console.log('Service health:', data);
            if (data.status === 'healthy') {
                Utils.showToast('Service is ready', 'success');
            }
        })
        .catch(error => {
            console.error('Health check failed:', error);
            Utils.showToast('Service is experiencing issues', 'danger');
        });
    
    // Add image validation
    const imageUpload = document.getElementById('imageUpload');
    if (imageUpload) {
        imageUpload.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Check file size (10MB limit)
                if (file.size > 10 * 1024 * 1024) {
                    Utils.showToast('File size exceeds 10MB limit', 'warning');
                    e.target.value = '';
                    return;
                }
                
                // Check file type
                const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
                if (!allowedTypes.includes(file.type)) {
                    Utils.showToast('Please select a valid image file (JPG, PNG, GIF, WebP)', 'warning');
                    e.target.value = '';
                    return;
                }
                
                Utils.showToast(`Selected: ${file.name} (${Utils.formatBytes(file.size)})`, 'info');
            }
        });
    }
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const submitBtn = document.getElementById('submitBtn');
            if (submitBtn && !submitBtn.disabled) {
                submitBtn.click();
            }
        }
        
        // Escape to clear form
        if (e.key === 'Escape') {
            const removeBtn = document.getElementById('removeImage');
            if (removeBtn) {
                removeBtn.click();
            }
        }
    });
});

// Export for use in browser console
window.ProductClassifier = {
    Utils,
    APIClient
};
