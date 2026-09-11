### Echelon Quant V1 — Repository Structure

```text
echelon-quant/
│
├── apps/
│   ├── frontend/                         # Web dashboard
│   │   ├── src/
│   │   │   ├── app/
│   │   │   ├── components/
│   │   │   ├── features/
│   │   │   │   ├── bots/
│   │   │   │   ├── accounts/
│   │   │   │   ├── positions/
│   │   │   │   ├── orders/
│   │   │   │   └── dashboard/
│   │   │   ├── services/
│   │   │   ├── hooks/
│   │   │   ├── types/
│   │   │   └── lib/
│   │   ├── public/
│   │   ├── package.json
│   │   └── tsconfig.json
│   │
│   └── api-gateway/                      # Main API entry point
│       ├── app/
│       │   ├── routes/
│       │   ├── middleware/
│       │   ├── schemas/
│       │   ├── services/
│       │   └── main.py
│       ├── tests/
│       ├── requirements.txt
│       └── Dockerfile
│
├── services/
│   │
│   ├── bot-service/                      # Bot creation/configuration
│   │   ├── app/
│   │   │   ├── api/
│   │   │   ├── models/
│   │   │   ├── schemas/
│   │   │   ├── services/
│   │   │   └── main.py
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   ├── account-service/                  # Deriv accounts & balances
│   │   ├── app/
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   ├── market-data-service/              # Live Deriv market data
│   │   ├── app/
│   │   │   ├── deriv/
│   │   │   ├── websocket/
│   │   │   ├── collectors/
│   │   │   ├── processors/
│   │   │   └── main.py
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   ├── analysis-service/                 # Market analysis/features
│   │   ├── app/
│   │   │   ├── indicators/
│   │   │   ├── features/
│   │   │   ├── volatility/
│   │   │   ├── market_structure/
│   │   │   ├── regime/
│   │   │   └── main.py
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   ├── strategy-ai-service/              # Strategies + ML prediction
│   │   ├── app/
│   │   │   ├── strategies/
│   │   │   │   ├── trend/
│   │   │   │   ├── breakout/
│   │   │   │   ├── momentum/
│   │   │   │   └── mean_reversion/
│   │   │   ├── models/
│   │   │   ├── prediction/
│   │   │   ├── training/
│   │   │   ├── feature_engineering/
│   │   │   └── main.py
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   ├── risk-service/                     # Hard risk-control layer
│   │   ├── app/
│   │   │   ├── rules/
│   │   │   ├── position_sizing/
│   │   │   ├── exposure/
│   │   │   ├── drawdown/
│   │   │   ├── limits/
│   │   │   └── main.py
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   ├── execution-service/                # Open/modify/close orders
│   │   ├── app/
│   │   │   ├── deriv/
│   │   │   ├── orders/
│   │   │   ├── execution/
│   │   │   ├── retry/
│   │   │   └── main.py
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   └── position-service/                 # Position management
│       ├── app/
│       │   ├── manager/
│       │   ├── trailing_stop/
│       │   ├── break_even/
│       │   ├── partial_close/
│       │   ├── pyramiding/
│       │   └── main.py
│       ├── tests/
│       ├── requirements.txt
│       └── Dockerfile
│
├── ml/
│   ├── datasets/                         # Training datasets
│   ├── notebooks/                        # Research
│   ├── experiments/
│   ├── training/
│   ├── evaluation/
│   ├── models/
│   └── configs/
│
├── backtesting/
│   ├── engine/
│   ├── strategies/
│   ├── data/
│   ├── metrics/
│   ├── reports/
│   └── tests/
│
├── shared/
│   ├── contracts/                        # Event/API contracts
│   ├── schemas/                          # Shared Pydantic schemas
│   ├── events/                           # Redis event definitions
│   ├── constants/
│   ├── exceptions/
│   ├── utils/
│   └── logging/
│
├── database/
│   ├── migrations/                       # Alembic
│   ├── seeds/
│   └── init/
│
├── infrastructure/
│   ├── docker/
│   │   ├── postgres/
│   │   └── redis/
│   ├── prometheus/
│   ├── grafana/
│   └── nginx/
│
├── config/
│   ├── development/
│   ├── testing/
│   └── production/
│
├── scripts/
│   ├── setup.sh
│   ├── start.sh
│   ├── stop.sh
│   ├── seed.py
│   └── train_model.py
│
├── tests/
│   ├── integration/
│   ├── e2e/
│   └── fixtures/
│
├── docs/
│   ├── architecture/
│   ├── api/
│   ├── trading/
│   ├── ai/
│   ├── risk-management/
│   └── development/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── tests.yml
│       └── build.yml
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── docker-compose.dev.yml
├── Makefile
├── README.md
└── LICENSE
```

### How the repository maps to the V1 architecture

```text
                         ECHELON QUANT V1
                                │
                         ┌──────▼──────┐
                         │   FRONTEND  │
                         │ apps/       │
                         │ frontend    │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │ API GATEWAY │
                         └──────┬──────┘
                                │
       ┌────────────────────────┼────────────────────────┐
       │                        │                        │
       ▼                        ▼                        ▼
 Bot Service             Account Service          Market Data
       │                        │                        │
       └────────────────────────┼────────────────────────┘
                                │
                                ▼
                       Analysis Service
                                │
                                ▼
                    Strategy + AI Service
                                │
                                ▼
                         Risk Service
                         ▲         │
                         │         ▼
                  ┌──────┴─── Execution Service
                  │                │
                  │                ▼
                  │           Deriv API
                  │                │
                  │                ▼
                  └──────── Position Service
                                   │
                                   ▼
                              PostgreSQL
```

### The important part

For V1, I would **not** make every folder a completely independent project. The services should share a common engineering standard:

```text
Service
 ├── api/
 ├── models/
 ├── schemas/
 ├── services/
 ├── repositories/
 ├── clients/
 ├── workers/
 ├── tests/
 └── main.py
```

And the trading decision should always follow:

```text
Market Data
     ↓
Analysis
     ↓
Strategy
     ↓
AI Prediction
     ↓
Signal
     ↓
RISK ENGINE  ← Hard safety barrier
     ↓
Execution
     ↓
Deriv
     ↓
Position Manager
```


