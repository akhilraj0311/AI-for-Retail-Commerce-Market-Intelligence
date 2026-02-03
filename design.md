RetailMind AI - Intelligent Commerce Decision Engine

## Document Information


## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Design](#2-architecture-design)
3. [Component Design](#3-component-design)
4. [Data Design](#4-data-design)
5. [AI/ML Pipeline Design](#5-aiml-pipeline-design)
6. [API Design](#6-api-design)
7. [User Interface Design](#7-user-interface-design)
8. [Security Design](#8-security-design)
9. [Deployment Architecture](#9-deployment-architecture)
10. [Technology Stack](#10-technology-stack)

## 1. System Overview

### 1.1 High-Level Architecture

RetailMind AI follows a **microservices-based architecture** with clear separation between:
- Frontend Layer: React-based web application
- API Gateway Layer: Centralized API management and routing
- Service Layer: Domain-specific microservices
- AI/ML Layer: Model serving and inference
- Data Layer: Multi-database architecture for different data types
- Integration Layer: Third-party API connectors


┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + Next.js)              │
│                 (Web Dashboard + AI Copilot UI)             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ HTTPS/WebSocket
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              API Gateway (Kong / AWS API Gateway)           │
│         (Authentication, Rate Limiting, Routing)            │
└───┬──────────────┬──────────────┬──────────────┬───────────┘
    │              │              │              │
    │              │              │              │
    ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Forecast │  │ Market   │  │   AI     │  │Inventory │
│ Service  │  │Intel Svc │  │ Copilot  │  │ Service  │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │
     │             │             │             │
     ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI/ML Infrastructure                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│   │ ML Models    │  │ LLM Service  │  │ Vector DB    │    │
│   │ (TensorFlow, │  │ (GPT-4,      │  │ (Pinecone/   │    │
│   │  PyTorch)    │  │  Gemini)     │  │  Weaviate)   │    │
│   └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
     │             │             │             │
     │             │             │             │
     ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │PostgreSQL│  │TimescaleDB│  │  Redis   │  │ MongoDB  │   │
│  │(Metadata)│  │(Time Series)│  │ (Cache)  │  │(Documents)│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
     │             │             │             │
     │             │             │             │
     ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│              Integration / External Services                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │E-commerce│  │Marketplace│  │Weather API│  │ Payment  │   │
│  │Platforms │  │APIs      │  │Economic   │  │Gateways  │   │
│  │(Shopify) │  │(Amazon)  │  │Data       │  │ (Stripe) │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘


### 1.2 Design Principles

1. **Microservices Architecture**: Independent, scalable services with single responsibility
2. **Event-Driven Communication**: Asynchronous messaging for loose coupling
3. **API-First Design**: All functionality exposed via well-documented APIs
4. **Cloud-Native**: Designed for containerized deployment and auto-scaling
5. **Security by Design**: Authentication, authorization, and encryption at every layer
6. **Data-Driven**: Analytics and ML at the core of decision-making
7. **Observability**: Comprehensive logging, monitoring, and tracing



## 2. Architecture Design

### 2.1 Microservices Architecture

#### 2.1.1 Service Decomposition

| Service Name | Responsibility | Technology Stack | Database |
|-------------|----------------|------------------|----------|
| **User Service** | Authentication, user management, roles | Node.js/Express | PostgreSQL |
| **Forecast Service** | Demand forecasting, model training | Python/FastAPI | TimescaleDB, S3 |
| **Market Intelligence Service** | Competitor tracking, price monitoring | Python/FastAPI | MongoDB, Redis |
| **AI Copilot Service** | NLP, LLM integration, RAG | Python/FastAPI | Pinecone, PostgreSQL |
| **Inventory Service** | Stock management, reorder logic | Node.js/NestJS | PostgreSQL |
| **Pricing Service** | Dynamic pricing, optimization | Python/FastAPI | PostgreSQL, Redis |
| **Analytics Service** | Reporting, dashboards, metrics | Python/FastAPI | ClickHouse |
| **Integration Service** | Third-party API management | Node.js/Express | MongoDB |
| **Notification Service** | Alerts, emails, SMS | Node.js/Express | Redis |

#### 2.1.2 Service Communication Patterns

**Synchronous Communication** (REST/GraphQL):
- Frontend ↔ API Gateway ↔ Services
- Inter-service direct calls (minimal, for immediate consistency)

**Asynchronous Communication** (Message Queue):
- Event-driven workflows
- Long-running operations (ML training, bulk imports)
- Notification distribution

**Message Broker**: RabbitMQ or Apache Kafka
- Topics: `forecast.generated`, `inventory.low`, `price.changed`, `alert.triggered`



### 2.2 API Gateway Design

**Technology**: Kong API Gateway or AWS API Gateway

**Responsibilities**:
- Request routing to appropriate microservices
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- API versioning
- Caching frequently accessed data
- CORS handling
- Analytics and logging

**Features**:
```yaml
# Example Kong Configuration
services:
  - name: forecast-service
    url: http://forecast-service:8000
    routes:
      - name: forecast-routes
        paths:
          - /api/v1/forecasts
    plugins:
      - name: jwt
      - name: rate-limiting
        config:
          minute: 60
          hour: 1000
      - name: cors
```


### 2.3 Data Flow Architecture

#### 2.3.1 Demand Forecasting Flow

```
User Upload → Data Validation → Feature Engineering → Model Selection
    ↓              ↓                   ↓                    ↓
CSV/API    Check schema,      Create features:      Choose best model
           data quality        - Lag features        based on product
                              - Rolling stats        characteristics
                              - Holiday flags
                              - Weather data
    ↓
Model Training/Inference → Post-processing → Store Results → Notify User
    ↓                          ↓                 ↓              ↓
Multiple models run      Apply business      TimescaleDB     Email/webhook
in parallel             constraints         + cache          + dashboard
                        (min/max values)                      update
```

#### 2.3.2 AI Copilot Query Flow

```
User Query → Intent Classification → Context Retrieval → LLM Inference
    ↓              ↓                        ↓                  ↓
"What's my      Determine query type:   Fetch relevant:    GPT-4 generates
 top product?"  - Data query            - Sales data       response using
                - Insight request       - Product info     retrieved context
                - Recommendation        - User history

    ↓
Response Formatting → Visualization Generation → Return to User
    ↓                        ↓                         ↓
Structure answer      Auto-create charts        JSON response
with citations       if data visualization      with text + viz
```

---

## 3. Component Design

### 3.1 Forecast Service

**Language**: Python 3.11+  
**Framework**: FastAPI  
**ML Libraries**: scikit-learn, XGBoost, Prophet, TensorFlow, PyTorch

#### 3.1.1 Component Architecture

```
forecast-service/
├── api/
│   ├── routes/
│   │   ├── forecast.py          # Forecast CRUD endpoints
│   │   ├── models.py            # Model management endpoints
│   │   └── training.py          # Training trigger endpoints
│   └── dependencies.py          # FastAPI dependencies
├── core/
│   ├── config.py                # Service configuration
│   ├── security.py              # Authentication utilities
│   └── database.py              # DB connection management
├── models/
│   ├── arima.py                 # ARIMA forecasting
│   ├── prophet.py               # Facebook Prophet
│   ├── xgboost_model.py         # XGBoost regressor
│   ├── lstm.py                  # LSTM neural network
│   └── ensemble.py              # Model ensembling
├── services/
│   ├── forecast_service.py      # Business logic
│   ├── feature_engineering.py   # Feature creation
│   ├── model_selection.py       # Auto model selection
│   └── evaluation.py            # Model evaluation metrics
├── schemas/
│   ├── forecast.py              # Pydantic models
│   └── training.py
├── workers/
│   ├── training_worker.py       # Async training jobs
│   └── batch_forecast.py        # Batch prediction
└── utils/
    ├── data_validation.py
    └── external_data.py         # Weather/holiday APIs
```

#### 3.1.2 Key Algorithms

**Time Series Models**:
```python
# ARIMA/SARIMA Implementation
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarima(data, order=(1,1,1), seasonal_order=(1,1,1,12)):
    model = SARIMAX(
        data, 
        order=order, 
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    return results

# Facebook Prophet
from prophet import Prophet

def fit_prophet(df):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model.add_country_holidays(country_name='US')
    model.fit(df)
    return model
```

**Machine Learning Models**:
```python
# XGBoost for demand forecasting
import xgboost as xgb

def train_xgboost(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'rmse'
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dval, 'validation')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    return model
```

**Ensemble Approach**:
```python
class EnsembleForecast:
    def __init__(self):
        self.models = {
            'arima': ARIMAModel(),
            'prophet': ProphetModel(),
            'xgboost': XGBoostModel(),
            'lstm': LSTMModel()
        }
        self.weights = None
    
    def fit(self, train_data, val_data):
        predictions = {}
        for name, model in self.models.items():
            model.fit(train_data)
            predictions[name] = model.predict(val_data)
        
        # Optimize weights based on validation performance
        self.weights = self._optimize_weights(predictions, val_data)
    
    def predict(self, data):
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(data)
        
        # Weighted average
        ensemble_pred = sum(
            self.weights[name] * pred 
            for name, pred in predictions.items()
        )
        return ensemble_pred
```

#### 3.1.3 Feature Engineering

```python
class FeatureEngineer:
    def create_features(self, df):
        df = df.copy()
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features
        for lag in [1, 7, 14, 30, 365]:
            df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'sales_rolling_mean_{window}'] = (
                df['sales'].rolling(window=window).mean()
            )
            df[f'sales_rolling_std_{window}'] = (
                df['sales'].rolling(window=window).std()
            )
        
        # Holiday indicator
        df['is_holiday'] = df['date'].isin(self.get_holidays())
        
        # External data (weather, events)
        df = self.add_weather_data(df)
        df = self.add_event_data(df)
        
        return df
```

---

### 3.2 AI Copilot Service

**Language**: Python 3.11+  
**Framework**: FastAPI  
**LLM Integration**: OpenAI GPT-4, Google Gemini Pro  
**Vector DB**: Pinecone or Weaviate

#### 3.2.1 RAG (Retrieval-Augmented Generation) Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query Input                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   Query Understanding       │
        │  - Intent classification    │
        │  - Entity extraction        │
        │  - Query rewriting          │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Semantic Search            │
        │  - Embed query              │
        │  - Search vector DB         │
        │  - Retrieve top K chunks    │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Context Assembly           │
        │  - Rank retrieved docs      │
        │  - Filter by relevance      │
        │  - Format for LLM           │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  LLM Inference              │
        │  - System prompt injection  │
        │  - Context + query          │
        │  - Generate response        │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Post-Processing            │
        │  - Format response          │
        │  - Add citations            │
        │  - Generate visualizations  │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │    Return to User           │
        └─────────────────────────────┘
```

#### 3.2.2 Implementation

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

class AICopilot:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Pinecone(
            index_name="retail-knowledge",
            embedding=self.embeddings
        )
        self.llm = OpenAI(model="gpt-4-turbo", temperature=0.2)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            return_source_documents=True
        )
    
    async def query(self, user_query: str, user_context: dict):
        # Enhance query with user context
        enhanced_query = self._enhance_query(user_query, user_context)
        
        # Classify intent
        intent = self._classify_intent(user_query)
        
        # Route based on intent
        if intent == "data_query":
            return await self._handle_data_query(enhanced_query)
        elif intent == "insight_request":
            return await self._handle_insight_request(enhanced_query)
        elif intent == "recommendation":
            return await self._handle_recommendation(enhanced_query)
        else:
            return await self._handle_general_query(enhanced_query)
    
    def _enhance_query(self, query: str, context: dict):
        system_prompt = f"""
        You are an AI assistant for RetailMind, a retail analytics platform.
        
        User Context:
        - Business: {context.get('business_name')}
        - Industry: {context.get('industry')}
        - Store Count: {context.get('store_count')}
        - Top Products: {context.get('top_products', [])[:5]}
        
        Answer the user's question using available data and provide
        actionable insights specific to their business.
        """
        return system_prompt + "\n\nUser Question: " + query
    
    async def _handle_data_query(self, query: str):
        # Execute SQL or query structured data
        result = await self.data_service.execute_query(query)
        
        # Format result for LLM
        formatted_data = self._format_data_for_llm(result)
        
        # Generate natural language response
        response = await self.llm.agenerate([
            f"Data: {formatted_data}\n\nQuestion: {query}\n\n"
            "Provide a clear, concise answer with key insights."
        ])
        
        return {
            "answer": response.generations[0][0].text,
            "data": result,
            "visualization": self._suggest_visualization(result)
        }
```

#### 3.2.3 Prompt Engineering

**System Prompts by Intent**:

```python
SYSTEM_PROMPTS = {
    "data_query": """
    You are a data analyst AI. Answer questions about sales, inventory,
    and business metrics. Always provide:
    1. Direct answer to the question
    2. Key insights from the data
    3. Trends or patterns observed
    4. Recommendations if applicable
    
    Use bullet points for clarity. Cite data sources.
    """,
    
    "recommendation": """
    You are a retail business consultant AI. Provide actionable
    recommendations based on:
    1. Historical data analysis
    2. Industry best practices
    3. Market trends
    4. Business context
    
    Structure recommendations as:
    - What to do
    - Why it matters
    - Expected impact
    - How to implement
    """,
    
    "insight_request": """
    You are an analytics AI specializing in retail insights. 
    Analyze data to identify:
    1. Anomalies and outliers
    2. Trends and seasonality
    3. Correlations and patterns
    4. Opportunities and risks
    
    Explain insights in business terms, not technical jargon.
    """
}
```

---

### 3.3 Market Intelligence Service

**Language**: Python 3.11+  
**Framework**: FastAPI  
**Scraping**: BeautifulSoup, Selenium, Playwright  
**Sentiment Analysis**: HuggingFace Transformers

#### 3.3.1 Web Scraping Architecture

```python
class CompetitorPriceMonitor:
    def __init__(self):
        self.scrapers = {
            'amazon': AmazonScraper(),
            'walmart': WalmartScraper(),
            'target': TargetScraper(),
            'generic': GenericScraper()
        }
        self.cache = Redis()
        self.db = MongoDB()
    
    async def monitor_product(self, product_url: str, frequency: str):
        # Determine scraper
        scraper = self._select_scraper(product_url)
        
        # Schedule periodic scraping
        schedule = self._create_schedule(frequency)
        
        # Add to monitoring queue
        await self.queue.publish({
            'url': product_url,
            'scraper': scraper,
            'schedule': schedule
        })
    
    async def scrape_price(self, url: str):
        # Check cache first
        cached = await self.cache.get(f"price:{url}")
        if cached and self._is_recent(cached):
            return cached
        
        # Scrape fresh data
        scraper = self._select_scraper(url)
        data = await scraper.scrape(url)
        
        # Store in DB and cache
        await self.db.prices.insert_one({
            'url': url,
            'price': data['price'],
            'in_stock': data['in_stock'],
            'timestamp': datetime.now(),
            'product_name': data['name']
        })
        await self.cache.setex(f"price:{url}", 3600, data)
        
        # Check for price changes and alert
        await self._check_price_alert(url, data)
        
        return data
```

#### 3.3.2 Sentiment Analysis

```python
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.aspect_model = pipeline(
            "text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
    
    def analyze_reviews(self, reviews: List[str]):
        # Overall sentiment
        sentiments = self.sentiment_pipeline(reviews)
        
        # Aspect-based sentiment
        aspects = ['quality', 'price', 'service', 'delivery']
        aspect_sentiments = {}
        
        for aspect in aspects:
            aspect_reviews = [
                r for r in reviews 
                if aspect.lower() in r.lower()
            ]
            if aspect_reviews:
                scores = self.sentiment_pipeline(aspect_reviews)
                aspect_sentiments[aspect] = self._aggregate_scores(scores)
        
        return {
            'overall_sentiment': self._aggregate_scores(sentiments),
            'aspect_sentiments': aspect_sentiments,
            'review_count': len(reviews)
        }
```

---

### 3.4 Inventory Service

**Language**: TypeScript/Node.js  
**Framework**: NestJS  
**Database**: PostgreSQL

#### 3.4.1 Reorder Logic

```typescript
class InventoryOptimizationService {
  async calculateReorderPoint(sku: string): Promise<ReorderPoint> {
    const product = await this.productRepo.findOne({ sku });
    const salesHistory = await this.getSalesHistory(sku, 90); // 90 days
    
    // Calculate average daily demand
    const avgDailyDemand = this.calculateAvgDemand(salesHistory);
    
    // Calculate demand variability (standard deviation)
    const demandStdDev = this.calculateStdDev(salesHistory);
    
    // Get lead time from supplier
    const leadTime = product.supplier.leadTimeDays;
    
    // Calculate safety stock (Z-score * std dev * sqrt(lead time))
    const serviceLevel = 0.95; // 95% service level
    const zScore = this.getZScore(serviceLevel); // 1.65 for 95%
    const safetyStock = Math.ceil(
      zScore * demandStdDev * Math.sqrt(leadTime)
    );
    
    // Reorder point = (Avg Daily Demand × Lead Time) + Safety Stock
    const reorderPoint = Math.ceil(
      (avgDailyDemand * leadTime) + safetyStock
    );
    
    // Economic Order Quantity (EOQ)
    const annualDemand = avgDailyDemand * 365;
    const orderingCost = product.orderingCost;
    const holdingCost = product.holdingCost;
    const eoq = Math.ceil(
      Math.sqrt((2 * annualDemand * orderingCost) / holdingCost)
    );
    
    return {
      sku,
      reorderPoint,
      safetyStock,
      eoq,
      avgDailyDemand,
      leadTime
    };
  }
  
  async checkInventoryLevels(): Promise<void> {
    const allProducts = await this.productRepo.find();
    
    for (const product of allProducts) {
      const currentStock = product.currentStock;
      const reorderPoint = await this.calculateReorderPoint(product.sku);
      
      if (currentStock <= reorderPoint.reorderPoint) {
        await this.notificationService.sendAlert({
          type: 'LOW_STOCK',
          sku: product.sku,
          currentStock,
          reorderPoint: reorderPoint.reorderPoint,
          suggestedOrderQty: reorderPoint.eoq
        });
      }
    }
  }
}
```

---

## 4. Data Design

### 4.1 Database Schema

#### 4.1.1 PostgreSQL (Relational Data)

**Users Table**:
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    business_name VARCHAR(255),
    industry VARCHAR(100),
    phone VARCHAR(50),
    role VARCHAR(50) DEFAULT 'owner',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    subscription_tier VARCHAR(50) DEFAULT 'free'
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_business ON users(business_name);
```

**Products Table**:
```sql
CREATE TABLE products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    sku VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    category VARCHAR(200),
    brand VARCHAR(200),
    current_price DECIMAL(10, 2),
    cost_price DECIMAL(10, 2),
    current_stock INTEGER DEFAULT 0,
    reorder_point INTEGER,
    safety_stock INTEGER,
    eoq INTEGER,
    supplier_id UUID,
    lead_time_days INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

CREATE INDEX idx_products_user ON products(user_id);
CREATE INDEX idx_products_sku ON products(sku);
CREATE INDEX idx_products_category ON products(category);
```

**Forecasts Table**:
```sql
CREATE TABLE forecasts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    product_id UUID REFERENCES products(id),
    forecast_date DATE NOT NULL,
    predicted_demand DECIMAL(10, 2),
    lower_bound DECIMAL(10, 2),
    upper_bound DECIMAL(10, 2),
    confidence_level DECIMAL(5, 2),
    model_used VARCHAR(100),
    model_accuracy DECIMAL(5, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_forecasts_product_date ON forecasts(product_id, forecast_date);
CREATE INDEX idx_forecasts_user ON forecasts(user_id);
```

#### 4.1.2 TimescaleDB (Time-Series Data)

**Sales Table** (Hypertable):
```sql
CREATE TABLE sales (
    time TIMESTAMPTZ NOT NULL,
    user_id UUID NOT NULL,
    product_id UUID NOT NULL,
    sku VARCHAR(100),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2),
    total_amount DECIMAL(10, 2),
    location VARCHAR(200),
    channel VARCHAR(50), -- 'online', 'store', 'marketplace'
    discount_amount DECIMAL(10, 2) DEFAULT 0,
    customer_id VARCHAR(100)
);

SELECT create_hypertable('sales', 'time');

CREATE INDEX idx_sales_user_time ON sales (user_id, time DESC);
CREATE INDEX idx_sales_product_time ON sales (product_id, time DESC);
CREATE INDEX idx_sales_sku ON sales (sku, time DESC);
```

**Continuous Aggregates** (Pre-computed metrics):
```sql
-- Daily sales aggregation
CREATE MATERIALIZED VIEW daily_sales
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS day,
    user_id,
    product_id,
    SUM(quantity) as total_quantity,
    SUM(total_amount) as total_revenue,
    COUNT(*) as transaction_count,
    AVG(unit_price) as avg_price
FROM sales
GROUP BY day, user_id, product_id;

-- Refresh policy
SELECT add_continuous_aggregate_policy('daily_sales',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

#### 4.1.3 MongoDB (Document Store)

**Market Intelligence Collection**:
```javascript
{
  _id: ObjectId("..."),
  productUrl: "https://amazon.com/...",
  competitorName: "Amazon",
  productName: "Sample Product",
  price: 29.99,
  currency: "USD",
  inStock: true,
  rating: 4.5,
  reviewCount: 1234,
  scrapedAt: ISODate("2026-01-31T10:00:00Z"),
  priceHistory: [
    { date: ISODate("2026-01-30"), price: 31.99 },
    { date: ISODate("2026-01-29"), price: 29.99 }
  ],
  metadata: {
    brand: "BrandName",
    category: "Electronics",
    features: ["Feature 1", "Feature 2"]
  }
}

// Indexes
db.marketIntelligence.createIndex({ productUrl: 1, scrapedAt: -1 });
db.marketIntelligence.createIndex({ competitorName: 1, category: 1 });
```

**Reviews Collection**:
```javascript
{
  _id: ObjectId("..."),
  productId: "uuid-here",
  source: "amazon",
  rating: 5,
  reviewText: "Great product!",
  reviewerName: "John D.",
  reviewDate: ISODate("2026-01-30"),
  sentiment: {
    overall: "positive",
    score: 0.92,
    aspects: {
      quality: "positive",
      price: "neutral",
      service: "positive"
    }
  },
  helpful: 45,
  verified: true
}
```

#### 4.1.4 Redis (Cache & Session Store)

**Cache Keys Structure**:
```
# User session
session:{user_id} → { token, expiresAt, metadata }

# Forecast cache
forecast:{product_id}:{date} → { demand, confidence, model }

# Price cache
price:{competitor}:{product_url} → { price, inStock, timestamp }

# Rate limiting
ratelimit:{user_id}:{endpoint} → counter with TTL

# Real-time alerts
alerts:{user_id} → list of pending alerts
```

---

### 4.2 Data Lake Architecture (for ML Training)

**AWS S3 / Google Cloud Storage**:
```
s3://retailmind-datalake/
├── raw/                          # Raw ingested data
│   ├── sales/
│   │   ├── year=2026/
│   │   │   ├── month=01/
│   │   │   │   ├── day=31/
│   │   │   │   │   └── sales_20260131.parquet
│   ├── products/
│   └── external/
│       ├── weather/
│       └── holidays/
├── processed/                    # Cleaned and transformed
│   ├── features/
│   │   ├── daily_features.parquet
│   │   └── product_features.parquet
│   └── training/
│       ├── train_data.parquet
│       └── validation_data.parquet
├── models/                       # Trained ML models
│   ├── demand_forecast/
│   │   ├── v1.0/
│   │   │   ├── model.pkl
│   │   │   ├── metadata.json
│   │   │   └── metrics.json
│   └── sentiment/
└── embeddings/                   # Vector embeddings
    └── product_embeddings.npy
```

---

## 5. AI/ML Pipeline Design

### 5.1 MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Data Ingestion Layer                       │
│    (Sales data, Product catalog, External APIs)             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline                    │
│  (Apache Airflow / Prefect for orchestration)               │
│    • Data validation (Great Expectations)                   │
│    • Feature extraction (lag, rolling, seasonality)         │
│    • External data enrichment (weather, holidays)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Training Pipeline                         │
│  • Experiment tracking (MLflow, Weights & Biases)           │
│  • Hyperparameter tuning (Optuna, Ray Tune)                │
│  • Cross-validation                                          │
│  • Model versioning                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Evaluation & Registry                     │
│  • Performance metrics (MAPE, RMSE, MAE)                    │
│  • Model comparison                                          │
│  • A/B testing framework                                     │
│  • Model registry (MLflow Model Registry)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Deployment                                │
│  • Containerization (Docker)                                 │
│  • Model serving (TensorFlow Serving, TorchServe, FastAPI) │
│  • Canary deployments                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Monitoring                                │
│  • Prediction logging                                        │
│  • Data drift detection (Evidently AI)                      │
│  • Model performance tracking                                │
│  • Automated retraining triggers                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Training Pipeline (Apache Airflow DAG)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'demand_forecast_training',
    default_args=default_args,
    description='Weekly model retraining',
    schedule_interval='0 2 * * 0',  # Every Sunday at 2 AM
    catchup=False
)

def extract_data(**context):
    """Extract sales data from TimescaleDB"""
    # Implementation
    pass

def validate_data(**context):
    """Validate data quality using Great Expectations"""
    # Implementation
    pass

def engineer_features(**context):
    """Create features for model training"""
    # Implementation
    pass

def train_models(**context):
    """Train multiple forecasting models"""
    # Implementation
    pass

def evaluate_models(**context):
    """Evaluate and compare model performance"""
    # Implementation
    pass

def deploy_best_model(**context):
    """Deploy the best performing model"""
    # Implementation
    pass

# Define task dependencies
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

feature_task = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag
)

evaluate_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='deploy_best_model',
    python_callable=deploy_best_model,
    dag=dag
)

# Set up task pipeline
extract_task >> validate_task >> feature_task >> train_task >> evaluate_task >> deploy_task
```

---

## 6. API Design

### 6.1 RESTful API Endpoints

**Base URL**: `https://api.retailmind.ai/v1`

#### Authentication Endpoints

```
POST   /auth/register           # Register new user
POST   /auth/login              # Login and get JWT token
POST   /auth/refresh            # Refresh access token
POST   /auth/logout             # Logout (invalidate token)
POST   /auth/forgot-password    # Request password reset
POST   /auth/reset-password     # Reset password with token
```

#### Forecast Endpoints

```
GET    /forecasts                      # List all forecasts
POST   /forecasts                      # Create new forecast
GET    /forecasts/{id}                 # Get forecast details
PUT    /forecasts/{id}                 # Update forecast
DELETE /forecasts/{id}                 # Delete forecast

POST   /forecasts/bulk                 # Bulk forecast generation
GET    /forecasts/product/{product_id} # Get forecasts for product
GET    /forecasts/accuracy             # Get accuracy metrics

# Example Request
POST /forecasts
{
  "product_ids": ["uuid-1", "uuid-2"],
  "forecast_horizon": 30,
  "model": "auto",  # auto-select best model
  "include_confidence": true
}

# Example Response
{
  "forecast_id": "uuid",
  "status": "completed",
  "forecasts": [
    {
      "product_id": "uuid-1",
      "sku": "SKU-001",
      "predictions": [
        {
          "date": "2026-02-01",
          "demand": 125,
          "lower_bound": 105,
          "upper_bound": 145,
          "confidence": 0.85
        }
      ],
      "model_used": "xgboost",
      "accuracy": 0.87
    }
  ]
}
```

#### AI Copilot Endpoints

```
POST   /copilot/query               # Send natural language query
GET    /copilot/conversations       # List conversation history
GET    /copilot/conversations/{id}  # Get conversation details
DELETE /copilot/conversations/{id}  # Delete conversation

POST   /copilot/feedback            # Provide feedback on response

# Example Request
POST /copilot/query
{
  "query": "What are my top 5 selling products this month?",
  "conversation_id": "uuid",  # optional, for context
  "include_visualization": true
}

# Example Response
{
  "response": {
    "text": "Based on sales data for January 2026, your top 5 products are:\n\n1. Product A - 1,234 units sold ($45,678 revenue)\n2. Product B - 987 units sold ($34,567 revenue)\n...",
    "data": [...],
    "visualization": {
      "type": "bar_chart",
      "config": {...}
    },
    "sources": ["sales_data_jan_2026"],
    "confidence": 0.95
  },
  "conversation_id": "uuid",
  "query_id": "uuid"
}
```

#### Market Intelligence Endpoints

```
GET    /market/competitors              # List tracked competitors
POST   /market/competitors              # Add competitor to track
DELETE /market/competitors/{id}         # Stop tracking competitor

GET    /market/prices                   # Get price comparisons
POST   /market/prices/alerts            # Set price alert
GET    /market/trends                   # Get market trends
GET    /market/sentiment/{product_id}   # Get sentiment analysis
```

#### Inventory Endpoints

```
GET    /inventory                       # List all inventory
POST   /inventory                       # Add inventory item
PUT    /inventory/{id}                  # Update inventory
GET    /inventory/low-stock             # Get low stock items
GET    /inventory/reorder-suggestions   # Get reorder recommendations

# Example Response
GET /inventory/reorder-suggestions
{
  "suggestions": [
    {
      "product_id": "uuid",
      "sku": "SKU-001",
      "current_stock": 45,
      "reorder_point": 50,
      "suggested_order_qty": 200,
      "lead_time_days": 7,
      "estimated_stockout_date": "2026-02-10",
      "priority": "high"
    }
  ]
}
```

### 6.2 WebSocket API (Real-Time Updates)

```
wss://api.retailmind.ai/v1/ws

# Connection
{
  "type": "authenticate",
  "token": "jwt_token"
}

# Subscribe to events
{
  "type": "subscribe",
  "channels": ["inventory_alerts", "price_changes", "forecast_updates"]
}

# Event notification
{
  "type": "inventory_alert",
  "data": {
    "product_id": "uuid",
    "sku": "SKU-001",
    "current_stock": 5,
    "alert_type": "critical_low"
  },
  "timestamp": "2026-01-31T10:30:00Z"
}
```

---

## 7. User Interface Design

### 7.1 Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│  [Logo] RetailMind        Search    [Alerts] [User Menu]    │
├──────┬──────────────────────────────────────────────────────┤
│      │                                                       │
│ Nav  │  ┌───────────────────────────────────────────┐      │
│      │  │  Key Metrics (Cards)                      │      │
│ -Home│  │  Revenue | Sales | Inventory | Forecast   │      │
│ -For-│  └───────────────────────────────────────────┘      │
│  cast│                                                       │
│ -Mark│  ┌─────────────────────┐ ┌─────────────────────┐   │
│  et  │  │  Sales Trend        │ │  Forecast vs Actual │   │
│ -Inv-│  │  (Line Chart)       │ │  (Comparison Chart) │   │
│  ento│  │                     │ │                     │   │
│  ry  │  └─────────────────────┘ └─────────────────────┘   │
│ -AI  │                                                       │
│  Cop-│  ┌───────────────────────────────────────────┐      │
│  ilot│  │  Top Products Table                       │      │
│ -Rep-│  │  SKU | Name | Sales | Stock | Forecast    │      │
│  orts│  └───────────────────────────────────────────┘      │
│      │                                                       │
└──────┴───────────────────────────────────────────────────────┘
```

### 7.2 AI Copilot Interface

```
┌─────────────────────────────────────────────────────────────┐
│  AI Business Copilot                        [Minimize] [X]  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Previous Conversations]                                   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ User: What are my best selling products?           │    │
│  │                                                     │    │
│  │ AI: Based on January 2026 sales:                   │    │
│  │ 1. Product A - 1,234 units                         │    │
│  │ 2. Product B - 987 units                           │    │
│  │ [Show Chart]                                       │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ User: Should I reorder Product A?                  │    │
│  │                                                     │    │
│  │ AI: Yes, I recommend ordering 500 units because:   │    │
│  │ • Current stock: 120 units                         │    │
│  │ • Forecast demand: 450 units/month                 │    │
│  │ • Lead time: 14 days                               │    │
│  │ [Create Order]                                     │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  [Type your question...]                      [Send] [🎤]   │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Component Library (React)

**Technology**: React 18+, TypeScript, Tailwind CSS, shadcn/ui

```typescript
// components/Dashboard/MetricCard.tsx
interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  trend?: 'up' | 'down';
  icon?: React.ReactNode;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  trend,
  icon
}) => {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {change !== undefined && (
          <p className={`text-xs ${trend === 'up' ? 'text-green-600' : 'text-red-600'}`}>
            {trend === 'up' ? '↑' : '↓'} {Math.abs(change)}% from last month
          </p>
        )}
      </CardContent>
    </Card>
  );
};

// components/Forecast/ForecastChart.tsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

export const ForecastChart: React.FC<{ data: ForecastData[] }> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual Sales" />
        <Line type="monotone" dataKey="forecast" stroke="#82ca9d" name="Forecast" />
        <Line type="monotone" dataKey="lowerBound" stroke="#ccc" strokeDasharray="5 5" />
        <Line type="monotone" dataKey="upperBound" stroke="#ccc" strokeDasharray="5 5" />
      </LineChart>
    </ResponsiveContainer>
  );
};
```

---

## 8. Security Design

### 8.1 Authentication Flow

```
User Login
    ↓
POST /auth/login { email, password }
    ↓
Verify credentials (bcrypt compare)
    ↓
Generate JWT tokens
    ├─ Access Token (15 min expiry)
    └─ Refresh Token (7 days expiry)
    ↓
Store refresh token in database
    ↓
Return tokens to client
    ↓
Client stores:
    ├─ Access token in memory
    └─ Refresh token in httpOnly cookie
    ↓
Client includes access token in Authorization header
    ↓
API Gateway validates JWT
    ↓
Forward request to service with user context
```

### 8.2 JWT Structure

```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user-uuid",
    "email": "user@example.com",
    "role": "owner",
    "business_id": "business-uuid",
    "permissions": ["read:forecasts", "write:forecasts", "read:inventory"],
    "iat": 1706698800,
    "exp": 1706699700
  }
}
```

### 8.3 Role-Based Access Control (RBAC)

```typescript
enum Permission {
  READ_FORECASTS = 'read:forecasts',
  WRITE_FORECASTS = 'write:forecasts',
  READ_INVENTORY = 'read:inventory',
  WRITE_INVENTORY = 'write:inventory',
  READ_MARKET_INTEL = 'read:market_intel',
  MANAGE_USERS = 'manage:users',
  ACCESS_COPILOT = 'access:copilot'
}

const ROLE_PERMISSIONS = {
  owner: [
    Permission.READ_FORECASTS,
    Permission.WRITE_FORECASTS,
    Permission.READ_INVENTORY,
    Permission.WRITE_INVENTORY,
    Permission.READ_MARKET_INTEL,
    Permission.MANAGE_USERS,
    Permission.ACCESS_COPILOT
  ],
  manager: [
    Permission.READ_FORECASTS,
    Permission.WRITE_FORECASTS,
    Permission.READ_INVENTORY,
    Permission.WRITE_INVENTORY,
    Permission.READ_MARKET_INTEL,
    Permission.ACCESS_COPILOT
  ],
  analyst: [
    Permission.READ_FORECASTS,
    Permission.READ_INVENTORY,
    Permission.READ_MARKET_INTEL,
    Permission.ACCESS_COPILOT
  ],
  viewer: [
    Permission.READ_FORECASTS,
    Permission.READ_INVENTORY
  ]
};
```

### 8.4 Data Encryption

**At Rest**:
- PostgreSQL: Transparent Data Encryption (TDE)
- S3: Server-Side Encryption (SSE-KMS)
- MongoDB: Encryption at Rest (WiredTiger)

**In Transit**:
- TLS 1.3 for all API communications
- Certificate pinning for mobile apps
- Encrypted WebSocket connections

**Sensitive Data**:
```python
from cryptography.fernet import Fernet

class SensitiveDataEncryption:
    def __init__(self, key: bytes):
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        return self.cipher.decrypt(encrypted_data.encode()).decode()

# Use AWS KMS or HashiCorp Vault for key management
```

---

## 9. Deployment Architecture

### 9.1 Kubernetes Deployment

```yaml
# forecast-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: forecast-service
  namespace: retailmind
spec:
  replicas: 3
  selector:
    matchLabels:
      app: forecast-service
  template:
    metadata:
      labels:
        app: forecast-service
    spec:
      containers:
      - name: forecast-service
        image: retailmind/forecast-service:v1.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: timescale-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: cache-credentials
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: forecast-service
  namespace: retailmind
spec:
  selector:
    app: forecast-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: forecast-service-hpa
  namespace: retailmind
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: forecast-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 9.2 Infrastructure as Code (Terraform)

```hcl
# main.tf
provider "aws" {
  region = "us-east-1"
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "retailmind-prod"
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  eks_managed_node_groups = {
    general = {
      desired_size = 3
      min_size     = 2
      max_size     = 10
      
      instance_types = ["t3.large"]
      capacity_type  = "ON_DEMAND"
    }
    
    ml_workers = {
      desired_size = 2
      min_size     = 1
      max_size     = 5
      
      instance_types = ["g4dn.xlarge"]  # GPU instances for ML
      capacity_type  = "SPOT"
    }
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "postgres" {
  identifier           = "retailmind-postgres"
  engine               = "postgres"
  engine_version       = "15.4"
  instance_class       = "db.r6g.xlarge"
  allocated_storage    = 100
  storage_encrypted    = true
  
  db_name  = "retailmind"
  username = var.db_username
  password = var.db_password
  
  multi_az = true
  backup_retention_period = 7
  
  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "retailmind-redis"
  engine               = "redis"
  node_type            = "cache.r6g.large"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]
}
```

### 9.3 CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest --cov=./src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build and push Docker image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/forecast-service:$IMAGE_TAG .
          docker push $ECR_REGISTRY/forecast-service:$IMAGE_TAG
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to Kubernetes
        uses: azure/k8s-deploy@v4
        with:
          manifests: |
            k8s/deployment.yaml
            k8s/service.yaml
          images: |
            ${{ secrets.ECR_REGISTRY }}/forecast-service:${{ github.sha }}
          kubectl-version: 'latest'
```

---

## 10. Technology Stack

### 10.1 Complete Technology Matrix

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18, Next.js 14, TypeScript | Web application |
| | Tailwind CSS, shadcn/ui | UI components |
| | Recharts, D3.js | Data visualization |
| | TanStack Query | API state management |
| | Zustand | Client state management |
| **Backend** | Python 3.11+, FastAPI | ML services |
| | Node.js 20+, NestJS | Business services |
| | GraphQL (Apollo Server) | Flexible API queries |
| **AI/ML** | PyTorch 2.0, TensorFlow 2.14 | Deep learning |
| | scikit-learn, XGBoost, LightGBM | Traditional ML |
| | Prophet, statsmodels | Time series |
| | HuggingFace Transformers | NLP models |
| | OpenAI GPT-4, Google Gemini | LLMs |
| | LangChain | LLM orchestration |
| | Pinecone / Weaviate | Vector database |
| **Databases** | PostgreSQL 15 | Relational data |
| | TimescaleDB | Time-series data |
| | MongoDB 7 | Document store |
| | Redis 7 | Cache & sessions |
| | ClickHouse | Analytics warehouse |
| **Message Queue** | RabbitMQ / Apache Kafka | Async messaging |
| **Orchestration** | Apache Airflow / Prefect | Workflow orchestration |
| | Kubernetes (EKS/GKE) | Container orchestration |
| **Monitoring** | Prometheus, Grafana | Metrics & dashboards |
| | DataDog / New Relic | APM |
| | Sentry | Error tracking |
| | ELK Stack | Log aggregation |
| **DevOps** | Docker, Kubernetes | Containerization |
| | Terraform | Infrastructure as Code |
| | GitHub Actions | CI/CD |
| | Helm | Kubernetes package manager |
| **Cloud** | AWS / GCP / Azure | Cloud infrastructure |
| | S3 / GCS | Object storage |
| | CloudFront / Cloud CDN | Content delivery |
| **Security** | Auth0 / Keycloak | Identity management |
| | HashiCorp Vault | Secrets management |
| | Let's Encrypt | SSL certificates |

### 10.2 Development Tools

- **IDE**: VSCode, PyCharm, Cursor
- **API Testing**: Postman, Insomnia, curl
- **Database Tools**: DBeaver, pgAdmin, MongoDB Compass
- **Version Control**: Git, GitHub
- **Collaboration**: Notion, Slack, Linear
- **Design**: Figma

---


