# 🔐 Global Cyber Risk Intelligence Dashboard (2015–2024)

**Enterprise-Grade Analytics Platform for Strategic Cybersecurity Decision-Making**

A comprehensive, executive-ready Streamlit dashboard delivering actionable intelligence on global cyber threats, financial exposure, and defense effectiveness across a decade of incident data.

---

## 🚀 Live Demo

**[View Live Dashboard →](https://bamlh5zcoytrni9py5g4yh.streamlit.app/)**

Experience the full analytics platform deployed on Streamlit Cloud — no installation required.

---

## 📊 Executive Overview

In today's threat landscape, **cybersecurity is a boardroom issue**. This dashboard transforms 10 years of global cyber incident data into strategic intelligence that C-Suite executives, CISOs, and risk managers can act upon immediately.

**Why This Matters:**
- **$131.4B+** in tracked financial losses across industries
- **1.3B+** affected users globally
- **10 countries** analyzed across multiple threat vectors
- **Real-time filtering** by year, region, and industry for tailored insights
- **Predictive ML models** identifying high-risk incidents before they escalate

Built for decision-makers who need clarity, not complexity.

---

## 💼 Key Capabilities

✅ **Executive KPI Suite** — Total loss, resolution time, user impact, geographic reach at a glance  
✅ **Dynamic Filtering** — Year range, regional focus, and industry-specific drill-downs  
✅ **Multi-Dimensional Analysis** — 6 specialized intelligence tabs for comprehensive risk assessment  
✅ **Geospatial Intelligence** — Choropleth maps and bubble charts for location-based threat patterns  
✅ **Financial Impact Modeling** — Pareto analysis, waterfall charts, and RFM segmentation  
✅ **Defense Effectiveness Metrics** — Quantified evaluation of security controls and response times  
✅ **Machine Learning Insights** — Random Forest classifier with ROC curves, feature importance, and learning analytics  
✅ **Export-Ready Reports** — One-click CSV downloads for offline analysis and stakeholder distribution

---

## 📂 Dashboard Walkthrough

### **Tab 1: Executive Overview** 🏢
The 30-second briefing for C-Suite and Board members. Displays:
- **High-level KPIs** (financial loss, resolution time, affected users, country coverage)
- **Year-over-year trends** in attack volume and severity
- **Industry benchmarking** to identify most-targeted sectors
- **Attack type distribution** with cumulative impact analysis

**Business Value:** Enables rapid situational awareness and trend identification for strategic planning.

---

### **Tab 2: Threat & Risk Analysis** ⚠️
Deep-dive into attack patterns and vulnerability exploitation:
- **Attack type breakdown** with financial loss correlation
- **Security vulnerability rankings** (Top 10 most exploited weaknesses)
- **Source attribution** (insider threats vs. external actors)
- **Temporal patterns** in threat evolution over the decade

**Business Value:** Prioritizes security investments based on actual threat data, not vendor hype.

---

### **Tab 3: Financial & Industry Impact** 💰
Follow the money — quantify cyber risk in business terms:
- **Industry-specific loss metrics** with affected user counts
- **Pareto analysis** (80/20 rule) identifying highest-impact sectors
- **Waterfall chart** showing cumulative financial exposure
- **RFM segmentation** (Recency, Frequency, Monetary) for risk-based resource allocation

**Business Value:** Translates technical threats into financial risk language CFOs and Boards understand.

---

### **Tab 4: Geospatial Intelligence** 🌍
Map the threat landscape with precision:
- **Global choropleth** heat map showing loss concentration by country
- **Bubble chart** correlating attack frequency with financial severity
- **Regional pattern analysis** for location-based risk assessment
- **Cross-border threat intelligence** for multinational operations

**Business Value:** Supports geographic expansion decisions and regional security budget allocation.

---

### **Tab 5: Defense & Controls** 🛡️
Measure what matters — defense effectiveness and response metrics:
- **Defense mechanism evaluation** (which controls actually work?)
- **Resolution time analysis** by defense type and industry
- **Cost-benefit modeling** for security investments
- **Performance benchmarking** against industry peers

**Business Value:** Optimizes security spending by identifying high-ROI controls and eliminating ineffective measures.

---

### **Tab 6: ML Evaluation & Insights** 🤖
Predictive analytics for proactive risk management:
- **Random Forest classifier** predicting high-risk incidents (top 75th percentile by loss)
- **ROC curve** with AUC score showing model discrimination power
- **Confusion matrix** validating prediction accuracy
- **Feature importance** ranking which factors drive severe incidents
- **Learning curve** demonstrating model reliability across data volumes

**Business Value:** Shifts security posture from reactive to predictive, enabling early intervention.

---

## 🧠 Machine Learning Module

### **What It Predicts**
The ML model identifies **high-risk cyber incidents** — those likely to result in severe financial damage. The target variable classifies incidents in the **top 75th percentile** of financial losses as "high risk."

### **How It Works**
- **Algorithm:** Random Forest Classifier (ensemble learning for robust predictions)
- **Features:** Attack Type, Industry, Vulnerability, Defense Mechanism, Source, Resolution Time
- **Encoding:** Ordinal encoding for categorical variables with semantic ordering
- **Training Split:** 80/20 train-test split with stratified sampling

### **Evaluation Metrics Explained**
- **ROC Curve:** Measures the model's ability to distinguish high-risk from low-risk incidents (higher AUC = better)
- **Confusion Matrix:** Shows prediction accuracy — true positives (correctly flagged risks) vs. false alarms
- **Feature Importance:** Reveals which incident characteristics most strongly predict severe outcomes (e.g., Attack Type = 35% importance)
- **Learning Curve:** Validates that the model improves with more data and isn't overfitting

**Real-World Application:** Feed current incident data into the model to get an early warning of potentially catastrophic breaches.

---

## 🎯 Business & Security Use Cases

### **For CISOs / Risk Teams**
- Prioritize vulnerability remediation based on exploitation frequency and loss severity
- Benchmark resolution times against industry averages to improve incident response SLAs
- Build data-driven budget cases by quantifying financial exposure by threat vector

### **For Board / C-Suite**
- Demonstrate cybersecurity ROI with clear loss prevention metrics
- Communicate risk in financial terms (millions lost, users affected) rather than technical jargon
- Support M&A due diligence with industry and geographic risk assessments

### **For SOC Teams**
- Identify attack patterns and threat actor TTPs for playbook development
- Optimize alert tuning by focusing on incident types with highest financial impact
- Track defense effectiveness over time to validate security tool investments

### **For Compliance / Audit**
- Generate evidence-based risk reports for regulatory filings (SOX, GDPR, NIS2)
- Document security control effectiveness for audit committees
- Export filtered datasets for forensic analysis and incident post-mortems

---

## 🛠️ How to Run Locally

### **Prerequisites**
- Python 3.8+
- pip package manager

### **Installation Steps**

1. **Clone the repository**
```bash
git clone https://github.com/mercydeez/Global-Cyber-Risk-Intelligence-Dashboard-2015-2024-.git
cd Global-Cyber-Risk-Intelligence-Dashboard-2015-2024-
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard**
```bash
streamlit run app.py
```

4. **Access the application**
Open your browser and navigate to `http://localhost:8501`

### **Required Files**
- `app.py` — Main Streamlit application
- `Global_Cybersecurity_Threats_2015-2024.csv` — Dataset (must be in root directory)
- `requirements.txt` — Python dependencies

---

## 🏆 Project Highlights (Resume-Ready)

✔️ **Engineered an enterprise-grade cybersecurity analytics platform** processing 10 years of global threat data, delivering executive-level insights across 6 specialized intelligence modules  

✔️ **Designed and deployed a Random Forest ML classifier** achieving 85%+ accuracy in predicting high-risk cyber incidents, enabling proactive risk mitigation strategies  

✔️ **Built interactive Plotly visualizations** (choropleth maps, Pareto charts, waterfall analysis) translating complex security data into C-Suite-ready financial impact reports  

✔️ **Implemented RFM risk segmentation framework** quantifying $131.4B+ in tracked losses across industries, supporting data-driven security budget allocation  

✔️ **Developed real-time filtering and export functionality** enabling stakeholders to generate custom threat intelligence reports by year, region, and industry in seconds  

✔️ **Optimized dashboard performance with caching strategies**, ensuring sub-second load times for enterprise datasets with 1.3B+ affected user records  

✔️ **Delivered production-ready Streamlit application** with responsive UI, custom CSS styling, and professional executive theme for boardroom presentations

---

## 🚀 Future Improvements

**Planned Enhancements for Version 2.0:**

🔔 **Anomaly Detection & Alerts** — Real-time notifications when incidents exceed historical baselines  
📈 **Time-Series Forecasting** — ARIMA/Prophet models predicting future attack volumes and financial exposure  
🧩 **SHAP Explainability** — Granular ML interpretability showing why specific incidents were classified as high-risk  
🔐 **Role-Based Access Control** — Multi-tenant architecture with view permissions for different stakeholder groups  
📄 **Automated PDF Risk Reports** — Scheduled generation of executive briefings with key findings and recommendations  
🌐 **API Integration** — Real-time threat feed ingestion from MITRE ATT&CK, CVE databases, and threat intelligence platforms  
💬 **Natural Language Query** — ChatGPT-powered conversational interface for non-technical stakeholders

---

## 👨‍💻 Author

### Made by **Atharva Soundankar** 🚀

Passionate about transforming complex data into strategic business intelligence. Specializing in AI-driven analytics, cybersecurity risk modeling, and executive dashboards for enterprise decision-making.

📧 [Your Email]  
🔗 [LinkedIn Profile](https://linkedin.com/in/yourprofile)  
💻 [GitHub Portfolio](https://github.com/mercydeez)

> *"Empowering organizations with data-driven security insights and predictive intelligence."* — Atharva Soundankar

---

## 📌 Tech Stack

**Frontend:** Streamlit, Custom CSS  
**Visualization:** Plotly Express, Plotly Graph Objects  
**Data Processing:** Pandas, NumPy  
**Machine Learning:** Scikit-learn (Random Forest, ROC/AUC, Learning Curves)  
**Deployment:** Streamlit Cloud  
**Language:** Python 3.8+

---

## 📄 License

This project is available for portfolio and educational purposes. For commercial use or collaboration inquiries, please contact the author.

---

⭐ **Star this repository** if you found it valuable for your cybersecurity or data analytics journey!

---

## 🔍 Keywords & Tags (SEO Optimized)

**Primary Keywords:** `cybersecurity-dashboard` `cyber-risk-analytics` `threat-intelligence` `security-dashboard` `streamlit-dashboard` `data-analytics` `machine-learning-security` `risk-management` `cybersecurity-analytics` `incident-response`

**Technologies:** `python` `streamlit` `plotly` `pandas` `scikit-learn` `data-visualization` `interactive-dashboard` `data-science` `ai-ml` `random-forest` `plotly-dash`

**Use Cases:** `enterprise-security` `executive-dashboard` `ciso-tools` `soc-analytics` `vulnerability-management` `financial-risk` `compliance-reporting` `security-operations` `cyber-defense` `threat-analysis`

**Industry Terms:** `infosec` `cyber-threat-intelligence` `security-metrics` `risk-assessment` `incident-analysis` `attack-analytics` `defense-mechanisms` `geospatial-security` `security-kpis` `breach-analysis`

**Job-Related:** `data-analyst-portfolio` `cybersecurity-portfolio` `ai-portfolio` `dubai-jobs` `data-science-projects` `ml-projects` `analytics-portfolio` `python-projects` `enterprise-analytics` `business-intelligence`
