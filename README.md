# AIOps Quality Project — ML Inference with Drift Detection

End-to-end MLOps система: FastAPI inference-сервіс із детектором дрейфу, задеплоєний через ArgoCD з Helm, моніторинг через Prometheus + Grafana, логування через Loki + Promtail, CI/CD через GitLab CI.

## Архітектура

```
┌──────────────┐     ┌──────────────────────────────────────────┐
│  GitLab CI   │     │          EKS Cluster                     │
│              │     │                                          │
│ retrain-model│     │  ┌────────────┐    ┌─────────────────┐   │
│ build-image  │────▶│  │  ArgoCD    │───▶│  inference-api  │   │
│ bump-helm    │     │  │ (auto-sync)│    │  (FastAPI)      │   │
└──────────────┘     │  └────────────┘    │   /predict      │   │
                     │                    │   /metrics       │   │
                     │                    │   /health        │   │
                     │                    └────────┬────────┘   │
                     │                             │            │
                     │  ┌──────────┐  ┌────────────▼─────────┐  │
                     │  │ Promtail │─▶│   Loki               │  │
                     │  │ (stdout) │  │   (log aggregation)  │  │
                     │  └──────────┘  └──────────────────────┘  │
                     │                                          │
                     │  ┌──────────────┐    ┌──────────────┐    │
                     │  │ Prometheus   │───▶│   Grafana     │   │
                     │  │ (scrape)     │    │  (dashboard)  │   │
                     │  └──────────────┘    └──────────────┘    │
                     └──────────────────────────────────────────┘
```

### Компоненти

| Компонент | Опис |
|-----------|------|
| **FastAPI** | Inference-сервіс з `/predict`, `/health`, `/metrics` ендпоінтами |
| **Drift Detector** | Rule-based перевірка на аномальні значення фіч |
| **Helm Chart** | Kubernetes deployment з Prometheus анотаціями |
| **ArgoCD** | GitOps auto-sync деплой з GitLab репозиторію |
| **Prometheus** | Збір метрик: requests/min, latency, drift events |
| **Grafana** | Дашборд для візуалізації метрик |
| **Loki + Promtail** | Збір та перегляд логів з подів |
| **GitLab CI** | Retrain → Build → Deploy пайплайн |

## Передумови

- EKS-кластер з `kubectl` налаштованим
- ArgoCD в namespace `infra-tools`
- kube-prometheus-stack (Prometheus + Grafana) в namespace `monitoring`
- AWS CLI з профілем `sergijrylskyj`
- Docker для побудови образів

## 1. Побудова та завантаження Docker-образу

```bash
# Авторизація в ECR
aws ecr get-login-password --profile sergijrylskyj --region eu-north-1 | \
  docker login --username AWS --password-stdin 277383786265.dkr.ecr.eu-north-1.amazonaws.com

# Побудова образу
docker build -t inference-api .

# Тегування та push
docker tag inference-api:latest 277383786265.dkr.ecr.eu-north-1.amazonaws.com/inference-api:v1
docker push 277383786265.dkr.ecr.eu-north-1.amazonaws.com/inference-api:v1
```

## 2. Деплой через ArgoCD

```bash
# Деплой Loki для логування
kubectl apply -f argocd/loki-stack.yaml

# Деплой inference-api
kubectl apply -f argocd/application.yaml

# Перевірити стан
kubectl get applications -n infra-tools
kubectl get pods -n serving
```

## 3. Тестування API

### Port-forward

```bash
kubectl port-forward svc/inference-api 8080:80 -n serving
```

### Тестовий запит

```bash
# Нормальний запит (без дрейфу)
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5]]}'

# Відповідь:
# {"predictions": [0, 1], "drift_detected": false}
```

```bash
# Запит з дрейфом (аномальне значення > 100)
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[500.0, 3.5, 1.4, 0.2]]}'

# Відповідь:
# {"predictions": [2], "drift_detected": true}
```

### Перевірити метрики

```bash
curl http://localhost:8080/metrics | grep inference
```

### Перевірити health

```bash
curl http://localhost:8080/health
# {"status": "ok"}
```

## 4. Перевірка логування

### Через kubectl logs

```bash
kubectl logs -n serving -l app=inference-api --tail=50
```

Очікуваний вивід:
```
2026-02-15 19:00:01 INFO Received predict request with 2 instances
2026-02-15 19:00:01 INFO Input data: [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5]]
2026-02-15 19:00:01 INFO Predictions: [0, 1], drift_detected: False
```

### Через Loki в Grafana

1. Відкрити Grafana: http://localhost:3000 (admin / prom-operator)
2. Перейти в **Explore** → обрати **Loki**
3. Запит: `{namespace="serving", app="inference-api"}`

## 5. Перевірка детектора дрейфу

```bash
# Надіслати запит з аномальним значенням
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[999.9, 0.1, 0.1, 0.1]]}'

# Перевірити логи — має бути WARNING: Drift detected
kubectl logs -n serving -l app=inference-api --tail=5
```

Метрика `drift_detected_total` збільшиться в Prometheus:
```promql
drift_detected_total
```

## 6. Grafana Dashboard

### Імпорт дашборду

1. Відкрити Grafana: http://localhost:3000
2. **Dashboards** → **Import**
3. Завантажити файл `grafana/dashboards.json`

### Метрики на дашборді

| Панель | PromQL |
|--------|--------|
| Requests/min | `rate(inference_requests_total[1m]) * 60` |
| Latency p95 | `histogram_quantile(0.95, rate(inference_request_latency_seconds_bucket[5m]))` |
| Drift Alerts | `drift_detected_total` |
| Drift Rate | `rate(drift_detected_total[1m]) * 60` |
| Total Requests | `inference_requests_total` |
| Avg Latency | `rate(inference_request_latency_seconds_sum[5m]) / rate(inference_request_latency_seconds_count[5m])` |

## 7. GitLab CI — Retrain Pipeline

### Як працює

`.gitlab-ci.yml` містить 3 стадії:

1. **retrain-model** — запускає `model/train.py`, генерує нову модель
2. **build-inference-image** — будує Docker-образ через Kaniko, пушить в ECR
3. **bump-helm-tag** — оновлює `helm/values.yaml` з новим тегом → ArgoCD підхоплює зміни

### CI/CD змінні (Settings → CI/CD → Variables)

| Змінна | Опис |
|--------|------|
| `AWS_ACCESS_KEY_ID` | AWS Access Key |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Key |
| `CI_GIT_TOKEN` | GitLab Personal Access Token (для push в repo) |

### Ручний запуск

GitLab → CI/CD → Pipelines → **Run pipeline** → обрати гілку `final-project`

### Як оновити модель

1. Змінити параметри в `model/train.py`
2. Запустити pipeline вручну або через push
3. CI перетренує модель → побудує новий образ → оновить Helm тег
4. ArgoCD автоматично задеплоїть нову версію

## Структура проєкту

```
aiops-quality-project/
├── app/
│   └── main.py                    # FastAPI inference-сервіс
│   └── requirements.txt
├── model/
│   └── train.py                   # Скрипт тренування моделі
│   └── requirements.txt
├── helm/
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       ├── deployment.yaml
│       └── service.yaml
├── argocd/
│   ├── application.yaml           # ArgoCD Application для inference-api
│   └── loki-stack.yaml            # ArgoCD Application для Loki + Promtail
├── grafana/
│   └── dashboards.json            # Grafana dashboard
├── prometheus/
│   └── additionalScrapeConfigs.yaml
├── .gitlab-ci.yml                 # CI/CD pipeline
├── Dockerfile
└── README.md
```

## Очищення

```bash
# Видалити ArgoCD Applications
kubectl delete application inference-api -n infra-tools
kubectl delete application loki-stack -n infra-tools

# Видалити namespace
kubectl delete namespace serving

# Видалити ECR repository
aws ecr delete-repository --repository-name inference-api --force --profile sergijrylskyj --region eu-north-1
```
