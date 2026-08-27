/**
 * Sample resume/JD pairs for the "Load example" button. Spread across the fit
 * spectrum so a demo of the scorer shows more than one part of the range.
 */
export interface Example {
  label: string;
  resume: string;
  jd: string;
}

export const EXAMPLES: Example[] = [
  {
    label: "Strong match — senior backend",
    resume: `SENIOR BACKEND ENGINEER

6 years building high-throughput services in Python. Django and FastAPI in
production, PostgreSQL, Redis, and heavy use of async I/O. Owned the payments
platform at a Series B fintech ($4M/day in transactions), led a team of 4,
ran the on-call rotation, and drove the migration from a Rails monolith to
event-driven microservices on AWS (ECS, SQS, RDS). Built the CI/CD pipeline
(GitHub Actions, blue/green deploys) and mentored two engineers to mid-level.

Skills: Python, FastAPI, Django, PostgreSQL, Redis, AWS, Docker, Kubernetes,
Terraform, CI/CD, distributed systems, observability (Datadog, OpenTelemetry).`,
    jd: `Senior Backend Engineer — Payments

We're hiring a senior backend engineer to own our payments platform. You'll
design and operate the services that move money for thousands of merchants.

Requirements:
- 4+ years of backend experience in Python
- Production experience with Django or FastAPI
- Strong relational database skills (PostgreSQL preferred)
- Cloud infrastructure experience, ideally AWS
- Comfort with Docker/Kubernetes and CI/CD pipelines
- A track record of technical leadership and mentoring

Nice to have: payments or fintech background, experience breaking up a
monolith, on-call ownership.`,
  },
  {
    label: "Solid match — missing one requirement",
    resume: `DATA ENGINEER

4 years building batch and streaming pipelines. Airflow, Spark (PySpark),
dbt, Snowflake. Built the company's event pipeline (Kafka -> Spark -> S3 ->
Snowflake) handling ~2B events/day. Strong SQL and Python. Some Terraform.

Skills: Python, SQL, Spark, Airflow, dbt, Snowflake, Kafka, AWS (S3, EMR).`,
    jd: `Senior Data Engineer

Build and own the data platform. Requirements: 4+ years data engineering,
strong Spark and SQL, orchestration (Airflow or Dagster), a cloud warehouse
(Snowflake/BigQuery/Redshift). You will also be responsible for our real-time
feature store built on Flink — prior stream-processing experience with Flink
or Kafka Streams is required, not just Kafka ingestion.`,
  },
  {
    label: "Partial match — adjacent role",
    resume: `DATA ANALYST

3 years. SQL (advanced), Python (pandas, matplotlib), Tableau and Looker
dashboards, A/B test design and analysis, weekly stakeholder reporting.
Built a churn-scoring model with scikit-learn that the CS team still uses.

Skills: SQL, Python, pandas, scikit-learn, Tableau, Looker, statistics, A/B testing.`,
    jd: `Machine Learning Engineer

5+ years building and shipping production ML systems. Deep PyTorch experience,
model serving (Triton, TorchServe, or similar), feature stores, distributed
training, and MLOps (CI/CD for models, monitoring, retraining). Strong
software engineering fundamentals in Python. You'll own models end to end,
from training infra to low-latency inference at scale.`,
  },
  {
    label: "Mismatch — different field",
    resume: `EXECUTIVE CHEF / RESTAURANT CO-OWNER

14 years in culinary operations. Menu development, food and labour cost
control, vendor negotiation, staff hiring and scheduling, health-code
compliance, catering for 300+ events. Grew a cafe-style restaurant's revenue
40% over three years.

Skills: menu planning, purchasing, P&L, team leadership, food safety.`,
    jd: `Senior iOS Engineer

8+ years of native iOS development in Swift and Objective-C. Shipped and
maintained 3+ apps at scale on the App Store. Deep knowledge of UIKit,
SwiftUI, Core Data, and performance profiling with Instruments. Experience
with CI for mobile (Fastlane, Xcode Cloud) and a strong eye for
architecture (MVVM, Composable Architecture).`,
  },
];
