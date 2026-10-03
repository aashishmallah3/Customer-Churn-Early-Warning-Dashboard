"""
LLM Retention Advisor Module:
Generates grounded retention advice (2-sentence summary + 1 concrete action)
based on customer attributes and SHAP top churn drivers.
Supports both direct API calls (Gemini/OpenAI) and a high-fidelity offline grounded engine.
"""

import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

def generate_retention_advice(
    customer_row: Dict[str, Any],
    top_drivers: List[Dict[str, Any]],
    churn_prob: float,
    api_key: Optional[str] = None,
    api_provider: str = "Google Gemini"
) -> Dict[str, str]:
    """
    Generates grounded retention advice for a specific customer.
    
    Returns:
        dict with keys:
            - 'summary': 2-sentence risk summary grounded in top drivers
            - 'action': 1 concrete retention action costing ~$20
            - 'source': 'API (Provider)' or 'Built-in Grounded Engine'
    """
    driver_names = [d["feature"] for d in top_drivers[:3]]
    driver_text = ", ".join([f"'{name}' (SHAP impact: +{d['shap_value']:.3f})" for name, d in zip(driver_names, top_drivers[:3])])
    
    # Check if user provided an API key and wants to call external LLM
    if api_key and api_key.strip():
        try:
            if "gemini" in api_provider.lower():
                return _call_gemini_api(customer_row, top_drivers, churn_prob, api_key.strip())
            elif "openai" in api_provider.lower():
                return _call_openai_api(customer_row, top_drivers, churn_prob, api_key.strip())
        except Exception as e:
            # Fall back gracefully with clear message
            fallback_result = _generate_rule_based_grounded_advice(customer_row, top_drivers, churn_prob)
            fallback_result["source"] = f"Built-in Grounded Engine (API error: {str(e)[:60]}...)"
            return fallback_result

    # Offline / Built-in Grounded Retention Engine
    return _generate_rule_based_grounded_advice(customer_row, top_drivers, churn_prob)


def _call_gemini_api(
    customer: Dict[str, Any],
    top_drivers: List[Dict[str, Any]],
    churn_prob: float,
    api_key: str
) -> Dict[str, str]:
    """Call Google Gemini API via lightweight urllib request."""
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    drivers_str = "\n".join([f"- {d['feature']}: SHAP contribution +{d['shap_value']:.3f}" for d in top_drivers[:3]])
    
    prompt = f"""You are a customer retention strategist for a telecommunications company.
Analyze this customer's churn risk and algorithmic drivers:

Customer Profile:
- Customer ID: {customer.get('customerID', 'N/A')}
- Churn Risk: {churn_prob * 100:.1f}%
- Tenure: {customer.get('tenure', 'N/A')} months
- Monthly Charges: ${customer.get('MonthlyCharges', 0):.2f}
- Contract Type: {customer.get('Contract', 'N/A')}
- Internet Service: {customer.get('InternetService', 'N/A')}
- Tech Support: {customer.get('TechSupport', 'N/A')}
- Payment Method: {customer.get('PaymentMethod', 'N/A')}

Top Algorithmic Churn Drivers (from SHAP TreeExplainer):
{drivers_str}

Task:
Respond strictly in JSON format with two keys:
1. "summary": Exactly a 2-sentence executive summary explaining why this specific customer is at risk, strictly grounded in the top drivers listed above.
2. "action": Exactly 1 concrete, realistic retention action (such as a promotion, discount, service add-on, or outreach costing around $20) that directly targets their primary churn driver.

Output valid JSON only with keys "summary" and "action". Do not include markdown code block formatting.
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
    }
    
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req, timeout=10) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        parsed = json.loads(raw_text)
        return {
            "summary": parsed.get("summary", ""),
            "action": parsed.get("action", ""),
            "source": "Google Gemini 1.5 Flash"
        }


def _call_openai_api(
    customer: Dict[str, Any],
    top_drivers: List[Dict[str, Any]],
    churn_prob: float,
    api_key: str
) -> Dict[str, str]:
    """Call OpenAI API via lightweight urllib request."""
    endpoint = "https://api.openai.com/v1/chat/completions"
    drivers_str = "\n".join([f"- {d['feature']}: SHAP contribution +{d['shap_value']:.3f}" for d in top_drivers[:3]])
    
    prompt = f"""You are a customer retention strategist for a telecommunications company.
Analyze this customer's churn risk and algorithmic drivers:

Customer Profile:
- Customer ID: {customer.get('customerID', 'N/A')}
- Churn Risk: {churn_prob * 100:.1f}%
- Tenure: {customer.get('tenure', 'N/A')} months
- Monthly Charges: ${customer.get('MonthlyCharges', 0):.2f}
- Contract Type: {customer.get('Contract', 'N/A')}
- Internet Service: {customer.get('InternetService', 'N/A')}
- Tech Support: {customer.get('TechSupport', 'N/A')}
- Payment Method: {customer.get('PaymentMethod', 'N/A')}

Top Algorithmic Churn Drivers (from SHAP TreeExplainer):
{drivers_str}

Respond strictly in JSON format with two keys:
1. "summary": Exactly a 2-sentence executive summary explaining why this specific customer is at risk, strictly grounded in the top drivers listed above.
2. "action": Exactly 1 concrete, realistic retention action costing around $20 that directly targets their primary churn driver.
"""

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a customer retention AI. Output only valid JSON with keys 'summary' and 'action'."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2
    }
    
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    )
    
    with urllib.request.urlopen(req, timeout=10) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        raw_text = res_data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(raw_text)
        return {
            "summary": parsed.get("summary", ""),
            "action": parsed.get("action", ""),
            "source": "OpenAI gpt-4o-mini"
        }


def _generate_rule_based_grounded_advice(
    customer: Dict[str, Any],
    top_drivers: List[Dict[str, Any]],
    churn_prob: float
) -> Dict[str, str]:
    """
    Deterministic, high-fidelity grounded retention engine.
    Produces guaranteed 2-sentence summaries and 1 targeted $20 action strictly tied to top SHAP drivers.
    """
    cid = customer.get("customerID", "Customer")
    tenure = customer.get("tenure", 0)
    contract = customer.get("Contract", "Month-to-month")
    charges = customer.get("MonthlyCharges", 0.0)
    internet = customer.get("InternetService", "DSL")
    support = customer.get("TechSupport", "No")
    payment = customer.get("PaymentMethod", "Electronic check")
    
    driver_names = [d["feature"].lower() for d in top_drivers[:3]]
    driver_display = [d["feature"].title() for d in top_drivers[:3]]
    
    # -------------------------------------------------------------
    # 1. Grounded 2-Sentence Summary
    # -------------------------------------------------------------
    sentence_1 = (
        f"Customer {cid} demonstrates an elevated {churn_prob*100:.1f}% churn risk, primarily driven by "
        f"their {driver_display[0]} and {driver_display[1]}."
    )
    
    if len(driver_display) >= 3:
        sentence_2 = (
            f"This vulnerability is further reinforced by {driver_display[2]}, "
            f"significantly lowering their long-term switching barrier."
        )
    else:
        sentence_2 = (
            f"Their ongoing engagement remains fragile without an active contractual or service commitment."
        )
    summary = f"{sentence_1} {sentence_2}"
    
    # -------------------------------------------------------------
    # 2. Targeted Retention Action (Budget: ~$20)
    # -------------------------------------------------------------
    primary_driver = driver_names[0] if driver_names else ""
    secondary_driver = driver_names[1] if len(driver_names) > 1 else ""
    combined_drivers = " ".join(driver_names)
    
    if "contract" in combined_drivers or contract == "Month-to-month":
        action = (
            f"Offer an exclusive 'Annual Loyalty Upgrade': provide a $20 one-time account statement credit "
            f"or $5/month bill reduction for 4 months in exchange for transitioning to a 12-month contract commitment."
        )
    elif "fiber" in combined_drivers or "monthlycharges" in combined_drivers or charges > 80:
        action = (
            f"Deploy a 'Speed & Value Retention Bundle': credit $20 against their next billing cycle "
            f"and lock in their current high-speed fiber pricing for 6 months to alleviate price sensitivity."
        )
    elif "techsupport" in combined_drivers or support == "No":
        action = (
            f"Provision a complimentary 6-month trial of '24/7 Premium Tech Concierge & Device Protection' "
            f"(valued at $18, internal cost $5), eliminating technical friction and establishing high operational stickiness."
        )
    elif "electronic check" in combined_drivers or "payment" in combined_drivers or payment == "Electronic check":
        action = (
            f"Present an 'Auto-Pay Security Credit': grant an immediate $20 credit upon enrolling in automatic "
            f"credit card or direct bank ACH payment to eliminate recurring manual bill payment churn."
        )
    elif "tenure" in combined_drivers or tenure <= 6:
        action = (
            f"Trigger a proactive 'VIP Welcome Check-in' outreach call from a tier-2 account specialist, accompanied "
            f"by a $20 streaming credit (e.g. Netflix/HBO add-on credit) to solidify early customer relationship tenure."
        )
    elif "onlinesecurity" in combined_drivers:
        action = (
            f"Activate a free 1-year 'CyberShield & Identity Protection' add-on ($20 retail value, marginal cost $3) "
            f"to deepen product dependency and elevate customer security satisfaction."
        )
    else:
        action = (
            f"Issue a personalized $20 customer appreciation bill credit alongside an invite to opt into "
            f"priority customer service support, directly targeting their {driver_display[0]}."
        )
        
    return {
        "summary": summary,
        "action": action,
        "source": "Built-in Grounded Retention Engine"
    }
