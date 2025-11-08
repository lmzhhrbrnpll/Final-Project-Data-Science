# E-Commerce Customer Churn Prediction Analysis

## Project Overview

This project analyzes customer churn patterns for an e-commerce company, identifying key factors that influence customer attrition and developing predictive models to flag at-risk customers before they leave.


## Business Problem

The e-commerce company faces a significant customer churn rate of 16.8%, leading to:
- Suboptimal revenue performance
- Operational inefficiencies
- Lost customer lifetime value
- Increased customer acquisition costs

## Project Objectives

### Primary Goals
- Identify key factors driving customer churn
- Build predictive models to estimate churn probability
- Develop actionable retention strategies
- Improve customer lifetime value

### Success Metrics
- Model Performance: Precision, Recall, F1-Score, ROC-AUC

## Key Insights Discovered

### Critical Churn Drivers
1. **Complaints**: Customers who filed complaints have significantly higher churn probability
2. **Tenure**: Shorter tenure correlates strongly with higher churn rates
3. **Product Categories**: Mobile phone category shows alarming 25-30% churn rate
4. **Payment Methods**: Cash-on-Delivery (COD) users have 29% churn rate
5. **Gender**: Male customers show slightly higher churn tendency

### Behavioral Patterns
- Customers with shorter membership duration are more likely to churn
- Distance from warehouse impacts retention (delivery speed concerns)
- Low satisfaction scores strongly predict imminent churn
- Infrequent shoppers show higher attrition rates

## Technical Approach

### Data Preprocessing
- Handled missing values and duplicates
- Outlier treatment using IQR method
- Categorical variable encoding
- Multicollinearity checking
- Train-test split (80:20 ratio)

### Model Development
**Algorithms Tested**:
- Random Forest
- Logistic Regression
- Gradient Boosting
- Support Vector Machines

**Final Model Selection**: Random Forest
- **F1-Score**: 91.59% (test set)
- **ROC-AUC**: 99%
- **Precision-Recall Balance**: Optimal performance
- **Train-Test Consistency**: Minimal performance gap

### Model Optimization
- Hyperparameter tuning (estimators, max_depth, min_samples_split, etc.)
- SMOTE evaluation (ultimately not used due to performance reduction)
- Feature importance analysis

## Model Performance

### Confusion Matrix Results
- **True Positives**: 940 customers correctly identified as churn risks
- **False Positives**: 1 customer incorrectly flagged
- **True Negatives**: 166 customers correctly predicted to stay
- **False Negatives**: 19 churning customers missed

### Key Strengths
- 99.50% AUC score indicates excellent classification capability
- Strong precision-recall balance
- Minimal overfitting between train and test sets

## Feature Importance Analysis

### Top Predictive Factors
1. **Tenure**: Longer tenure = higher loyalty
2. **WarehouseToHome**: Distance impacts delivery satisfaction
3. **SatisfactionScore**: Low scores strongly predict churn
4. **Complain**: Complaint history is a major red flag
5. **DaySinceLastOrder**: Purchase recency matters

## Strategic Recommendations

### Immediate Actions
1. **Targeted Cashback & Delivery Discounts**
   - Implement strategic cashback increases for high-risk customers
   - Offer shipping discounts to offset warehouse distance concerns

2. **Proactive Churn Prevention**
   - Create personalized offers for predicted churn risks
   - Assign dedicated customer service for high-value at-risk customers
   - Implement early warning system for satisfaction score drops

3. **Category-Specific Loyalty Programs**
   - Develop targeted rewards for mobile phone category shoppers
   - Create personalized recommendations based on preferred categories
   - Implement category-specific engagement campaigns

### Long-term Initiatives
1. **Customer Experience Enhancement**
   - Reduce delivery times for distant customers
   - Improve complaint resolution processes
   - Enhance mobile app user experience

2. **Loyalty Program Optimization**
   - Tiered membership benefits based on tenure
   - Personalized loyalty rewards
   - Progressive cashback structures

3. **Payment Method Diversification**
   - Incentivize digital payment adoption
   - Reduce dependency on COD payments
   - Educate customers on payment security

## Business Impact Expected

### Quantitative Benefits
- Potential churn reduction from 16.8% to under 12%
- Improved customer lifetime value
- Better resource allocation for retention efforts
- Enhanced customer satisfaction scores

### Operational Improvements
- Data-driven decision making for marketing campaigns
- Proactive rather than reactive customer retention
- Optimized customer service resource allocation

### Additional Link
- [Streamlit](https://ecomm-churn-pred.streamlit.app)
