# Medical Triage API — FIAP Tech Challenge Fase 3

Projeto acadêmico evolutivo para classificação da urgência de textos médicos. A entrega é
construída em quatro steps independentes, cada um preservando o que já funciona e adicionando
uma nova capacidade operacional.

Cada pasta é um snapshot executável da evolução. Ao iniciar um novo step, a implementação do
step anterior será copiada e evoluída dentro da pasta seguinte. Assim, `step1/` continuará
demonstrável mesmo depois da conclusão de `step2/`, `step3/` e `step4/`.

> **Aviso:** este projeto é exclusivamente educacional. O classificador não foi validado para
> uso clínico e não deve apoiar decisões de atendimento reais.

## Evolução do projeto

| Step | Estado | Evolução principal | Evidência esperada |
|---|---|---|---|
| Step 1 | **Concluído** | Decisão de arquitetura, API FastAPI, Docker e latência baseline | API containerizada e benchmark local |
| Step 2 | Planejado | GitHub Actions e pipeline de retreino com Airflow | Workflow verde e DAG funcional |
| Step 3 | Planejado | Prometheus, Grafana e Docker Compose | Dashboard com requisições, latência e erros |
| Step 4 | Planejado | Modelo NLP treinado e otimização ONNX | Comparação original versus otimizado e vídeo STAR |

Os commits devem seguir Conventional Commits com escopo por etapa:

```text
feat(step1): create initial triage API
fix(step2): make retraining task idempotent
test(step3): cover request metrics
docs(step4): add ONNX benchmark results
```

## Step 1 — Decisão arquitetural e API inicial

### Decisão: inferência real-time na AWS

A triagem precisa responder durante o atendimento. Por isso, o caminho principal é **real-time e
síncrono**, e não batch. Processamento batch continua adequado ao treino, retreino e avaliações
offline, mas não ao endpoint usado para priorizar um novo caso.

A arquitetura de produção proposta é:

```text
Client
  -> HTTPS / Application Load Balancer
  -> FastAPI container / Amazon ECS on AWS Fargate
  -> Versioned model artifact / Amazon S3

Container images -> Amazon ECR
Logs and alarms  -> Amazon CloudWatch
Secrets          -> AWS Secrets Manager
Retraining       -> Airflow batch pipeline -> S3 model artifact
```

#### Por que ECS com Fargate

- Reutiliza o mesmo container validado localmente, reduzindo diferenças entre ambientes.
- Mantém instâncias da API prontas, evitando que cold starts comprometam a latência de triagem.
- Elimina a administração direta de servidores EC2.
- Integra-se ao Application Load Balancer para health checks, HTTPS e distribuição de tráfego.
- Permite escalar tarefas por CPU, memória ou requisições por target.

Para um cenário real, as tarefas ficariam em sub-redes privadas, o ALB seria o único componente
exposto, e o texto clínico não seria persistido em logs. A infraestrutura em nuvem é uma decisão
documental nesta fase; o entregável executável continua sendo local.

Referências oficiais:

- [AWS — Load balancing para serviços ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-load-balancing.html)
- [AWS — Imagens de container para ECS e ECR](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/create-container-image.html)
- [AWS — ECS Service Auto Scaling](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-auto-scaling.html)

### Classificador provisório

O Step 1 usa `RuleBasedTriageClassifier`, uma implementação determinística pequena que classifica
sinais textuais em `normal`, `attention` ou `urgent`. Ela existe para validar API, Docker e medição
de latência antes do treinamento.

O contrato `TriageClassifier` desacopla a API da implementação. No Step 4, o classificador
heurístico será substituído pelo pipeline Scikit-Learn/ONNX sem alterar o endpoint `/predict`.

### Endpoints

| Método | Endpoint | Uso |
|---|---|---|
| `GET` | `/health` | Saúde e versões do serviço/classificador |
| `POST` | `/predict` | Classificação do texto por urgência |
| `GET` | `/docs` | Documentação OpenAPI interativa |

Exemplo:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"report":"Patient reports severe chest pain and difficulty breathing."}'
```

Resposta esperada:

```json
{
  "urgency": "urgent",
  "classifier_version": "step1-rule-based-v1",
  "processing_time_ms": 0.01
}
```

## Dataset recomendado

O dataset principal recomendado é o
[MIMIC-IV-ED v2.2 no PhysioNet](https://physionet.org/content/mimic-iv-ed/2.2/).
Ele possui aproximadamente 425 mil passagens por emergência. A tabela `triage` contém:

- `chiefcomplaint`: texto livre informado na triagem;
- `acuity`: severidade de 1, mais grave, até 5, menos grave.

### Como obter

1. Criar uma conta no PhysioNet.
2. Solicitar o status de usuário credenciado.
3. Concluir o treinamento `CITI Data or Specimens Only Research`.
4. Assinar o Data Use Agreement do MIMIC-IV-ED.
5. Baixar a tabela `triage` da versão 2.2 e mantê-la em `data/raw/`, que não é versionada.

Mapeamento inicial a ser validado metodologicamente no Step 4:

| Acuity original | Classe do projeto |
|---|---|
| 1–2 | `urgent` |
| 3 | `attention` |
| 4–5 | `normal` |

Esse agrupamento é uma simplificação acadêmica e não representa um protocolo clínico validado.

O [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
citado no enunciado é mais fácil de baixar e contém 14.438 textos, mas suas cinco classes são
categorias de doenças, não níveis de urgência. Ele pode servir para experimentar o pipeline, mas
não é a recomendação para a entrega final de triagem.

## Execução local

### Com Docker

```bash
docker build -t medical-triage-api:step1 step1
docker run --rm --name medical-triage-api -p 8000:8000 medical-triage-api:step1
```

Acesse `http://localhost:8000/docs`.

### Sem Docker

```bash
cd step1
uv sync
uv run uvicorn triage_api.main:app --app-dir src --reload
```

### Qualidade

```bash
cd step1
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Baseline de latência

Com o container em execução:

```bash
cd step1
uv run python scripts/benchmark.py \
  --warmup 20 \
  --requests 200 \
  --output benchmarks/step1-baseline.json
```

O benchmark mede o ciclo HTTP completo a partir da máquina host. Os resultados registrados nesta
máquina ficam em `benchmarks/step1-baseline.json`; eles são uma referência local e não um SLA.

<!-- STEP1_BENCHMARK_RESULTS -->

Resultado registrado em 26/08/2026 usando Windows 11, Docker Desktop 29.7.2 e o container
Python 3.12. Foram feitas 50 chamadas de aquecimento e 500 chamadas sequenciais medidas.

| Métrica | Latência |
|---|---:|
| Mínima | 1,626 ms |
| Média | 8,761 ms |
| Mediana | 2,470 ms |
| p95 | 26,056 ms |
| Máxima | 28,503 ms |

A mediana representa melhor a chamada típica nesta máquina; a diferença para a média e o p95
mostra a variação introduzida pelo ciclo HTTP e pela virtualização do Docker Desktop. O Step 4
reutilizará o mesmo script e ambiente para comparar o modelo original com o ONNX.

## Estrutura atual

```text
.
├── step1/                  # API inicial, Docker e baseline
│   ├── benchmarks/
│   ├── scripts/benchmark.py
│   ├── src/triage_api/
│   ├── tests/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── uv.lock
├── step2/                  # Próximo snapshot: CI/CD e Airflow
├── step3/                  # Próximo snapshot: Prometheus e Grafana
├── step4/                  # Próximo snapshot: modelo final e ONNX
└── README.md               # Índice e histórico da evolução
```

## Próximo incremento

O Step 2 adicionará lint e testes no GitHub Actions e uma DAG Airflow para ingestão, treino e
persistência do artefato, mantendo o mesmo contrato HTTP criado aqui.
