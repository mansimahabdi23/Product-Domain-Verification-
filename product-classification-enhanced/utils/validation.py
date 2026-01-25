import re
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    """Validates user inputs"""
    
    def __init__(self):
        # Domain text validation pattern
        self.domain_pattern = re.compile(r'^[a-zA-Z0-9\s,\.\-]+$')
        
        # Company name validation pattern
        self.company_pattern = re.compile(r'^[a-zA-Z0-9\s\-\&\.]+$')
    
    def validate_domain_text(self, domain_text: str) -> Tuple[bool, Optional[str]]:
        """Validate domain text input"""
        if not domain_text or not domain_text.strip():
            return False, "Domain text cannot be empty"
        
        domain_text = domain_text.strip()
        
        # Check length
        if len(domain_text) > 500:
            return False, "Domain text too long (max 500 characters)"
        
        # Check content
        if not self.domain_pattern.match(domain_text):
            return False, "Domain text contains invalid characters"
        
        # Check if it has meaningful content (at least 2 words)
        words = [w for w in domain_text.split() if len(w) > 2]
        if len(words) < 2:
            return False, "Domain text should contain at least 2 meaningful words"
        
        return True, None
    
    def validate_company_name(self, company_name: str) -> Tuple[bool, Optional[str]]:
        """Validate company name"""
        if not company_name:
            return True, None  # Company name is optional
        
        company_name = company_name.strip()
        
        if len(company_name) > 100:
            return False, "Company name too long (max 100 characters)"
        
        if not self.company_pattern.match(company_name):
            return False, "Company name contains invalid characters"
        
        return True, None
    
    def validate_top_k(self, top_k: Any) -> Tuple[bool, Optional[str]]:
        """Validate top_k parameter"""
        try:
            top_k = int(top_k)
            if not 1 <= top_k <= 20:
                return False, "top_k must be between 1 and 20"
            return True, None
        except (ValueError, TypeError):
            return False, "top_k must be an integer"
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize input text"""
        if not text:
            return ""
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove potentially harmful characters
        text = re.sub(r'[<>"\']', '', text)
        
        return text


# Global instance
validator = InputValidator()
