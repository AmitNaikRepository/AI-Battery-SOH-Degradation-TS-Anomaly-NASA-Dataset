"""
State of Health (SOH) Estimation Module
Calculates SOH from predicted capacity
"""

import numpy as np


class SOHEstimator:
    """Estimate battery State of Health"""
    
    def __init__(self, initial_capacity=1.9):
        """
        Initialize estimator
        
        Args:
            initial_capacity: Battery capacity when new (Ah)
        """
        self.initial_capacity = initial_capacity
        self.eol_threshold = 0.80  # 80% SOH = End of Life
    
    def calculate_soh(self, current_capacity):
        """
        Calculate SOH percentage
        
        Args:
            current_capacity: Current capacity (Ah)
            
        Returns:
            SOH percentage (0-100)
        """
        soh = (current_capacity / self.initial_capacity) * 100
        return max(0, min(100, soh))  # Clamp between 0-100
    
    def classify_health(self, soh):
        """
        Classify battery health status
        
        Args:
            soh: SOH percentage
            
        Returns:
            tuple (status, emoji, color)
        """
        if soh >= 95:
            return "Excellent", "🟢", "green"
        elif soh >= 85:
            return "Good", "🟢", "green"
        elif soh >= 75:
            return "Fair", "🟡", "yellow"
        elif soh >= 65:
            return "Poor", "🟠", "orange"
        else:
            return "Critical", "🔴", "red"
    
    def estimate_rul(self, soh, degradation_rate):
        """
        Estimate Remaining Useful Life
        
        Args:
            soh: Current SOH (%)
            degradation_rate: % loss per cycle
            
        Returns:
            Estimated cycles until EOL (80% SOH)
        """
        if degradation_rate <= 0:
            return float('inf')  # No degradation
        
        eol_soh = self.eol_threshold * 100
        remaining_soh = soh - eol_soh
        
        if remaining_soh <= 0:
            return 0  # Already below EOL
        
        rul_cycles = remaining_soh / degradation_rate
        return max(0, min(1000, rul_cycles))  # Cap at 1000 cycles
    
    def estimate_degradation_rate(self, soh, cycle):
        """
        Estimate degradation rate (% per cycle)
        
        Args:
            soh: Current SOH (%)
            cycle: Current cycle number
            
        Returns:
            Degradation rate (% per cycle)
        """
        if cycle == 0:
            return 0.0
        
        total_degradation = 100 - soh
        rate = total_degradation / cycle
        return rate
    
    def get_recommendation(self, soh, rul_cycles):
        """
        Get maintenance recommendation
        
        Args:
            soh: Current SOH (%)
            rul_cycles: Remaining useful life (cycles)
            
        Returns:
            Recommendation string
        """
        if soh >= 85:
            return "Battery in good condition. Continue normal operation."
        elif soh >= 75:
            return f"Monitor battery closely. Estimated {int(rul_cycles)} cycles remaining."
        elif soh >= 65:
            return f"Plan replacement soon. Estimated {int(rul_cycles)} cycles remaining."
        else:
            return "Battery near end of life. Replace immediately!"
    
    def estimate_full(self, predicted_capacity, cycle):
        """
        Complete SOH estimation
        
        Args:
            predicted_capacity: Predicted capacity (Ah)
            cycle: Current cycle number
            
        Returns:
            dict with all SOH metrics
        """
        soh = self.calculate_soh(predicted_capacity)
        status, emoji, color = self.classify_health(soh)
        degradation_rate = self.estimate_degradation_rate(soh, cycle)
        rul_cycles = self.estimate_rul(soh, degradation_rate)
        recommendation = self.get_recommendation(soh, rul_cycles)
        
        return {
            'predicted_capacity': float(predicted_capacity),
            'soh_percent': float(soh),
            'health_status': status,
            'health_emoji': emoji,
            'health_color': color,
            'degradation_rate': float(degradation_rate),
            'rul_cycles': float(rul_cycles),
            'eol_threshold': self.eol_threshold * 100,
            'recommendation': recommendation,
            'warranty_status': 'Valid' if soh >= 80 else 'Expired'
        }


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    estimator = SOHEstimator(initial_capacity=1.9)
    
    # Example prediction
    predicted_capacity = 1.65
    cycle = 100
    
    result = estimator.estimate_full(predicted_capacity, cycle)
    
    print("SOH Estimation Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")