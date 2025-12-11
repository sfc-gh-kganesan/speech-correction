# Snowflake Terms FAQ

This FAQ explains key Snowflake concepts and features listed in the provided YAML glossary, in clear, practical language. Each entry maps directly to a term from that list. For authoritative details, consult Snowflake product documentation and your internal guidelines. 

---

## Core Platform Objects

Q: What is an **Account**?  

A: A Snowflake account is your top-level tenancy where you manage users, roles, compute, and data objects (databases, schemas, tables) and configure governance and integrations. 

Q: What is an **Organization**?  

A: An organization groups multiple Snowflake accounts under unified governance, billing, and management; it's where org-level administration like ORGADMIN operates. 

Q: What is a **Region**?  

A: A cloud region (e.g., AWS, Azure, GCP) where your Snowflake account and data physically reside; it impacts data locality, latency, and cross-region features. 

Q: What is a **Virtual warehouse**?  

A: The compute layer in Snowflake; elastic clusters you size and auto-suspend/resume to run SQL, transformations, and services-based workloads. 

Q: What is a **Database**?  

A: A top-level container for schemas and data objects (tables, views, stages) that provides logical isolation and governance boundaries. 

Q: What is a **Schema**?  

A: A namespace within a database that organizes related objects (tables, views, functions) and helps manage privileges and structure. 

Q: What are **Tables** (permanent; transient; temporary)?  

A: Tables store data; permanent tables support full Time Travel and Fail-safe; transient reduce retention/cost; temporary are session-scoped and ephemeral. 

Q: What are **Views** (standard; secure)?  

A: Views are logical query definitions over data; secure views hide the underlying definition and enforce stronger data security for sharing. 

Q: What is a **Materialized view**?  

A: A view whose results are physically stored and maintained for faster reads; Snowflake handles incremental refresh based on data changes. 

Q: What is a **Sequence**?  

A: An object that generates unique increasing numbers, commonly used for surrogate keys or ordering. 

Q: What is a **Stage** (internal; external)?  

A: A staging area for files to load/unload; internal stages reside inside Snowflake; external stages reference cloud storage like S3, ADLS, or GCS. 

Q: What is a **File format**?  

A: A reusable definition for parsing staged files (e.g., JSON, Parquet, Avro) including delimiters, compression, and data-type options. 

Q: What is a **Pipe**?  

A: A Snowpipe object that continuously ingests new files from a stage, typically triggered by cloud notifications for near‑real‑time loads. 

Q: What is a **Stream**?  

A: A change data capture (CDC) object tracking inserts/updates/deletes on a table to drive incremental pipelines and materializations. 

Q: What is a **Task**?  

A: A scheduled or event-driven job that runs SQL (or DAGs of tasks) for orchestrating pipelines and refreshes. 

Q: What is a **Dynamic table**?  

A: A declarative, incrementally maintained table that stays up to date with source changes, simplifying near‑real‑time transformations. 

Q: What is an **External table**?  

A: A table that exposes data stored outside Snowflake (e.g., S3) without fully loading it, using metadata for schema-on-read. 

Q: What are **Apache Iceberg tables** (Snowflake-managed; externally managed)?  

A: Iceberg-format tables you can manage natively in Snowflake (Snowflake-managed catalog) or connect to external catalogs for interoperability. 

Q: What are **Hybrid tables (Unistore)**?  

A: Row‑oriented tables optimized for transactional/operational workloads alongside analytics, enabling mixed workloads in one platform. 

Q: What is a **Tag**?  

A: A governance label you attach to objects for classification, lineage, and policy enforcement (e.g., sensitivity tags). 

Q: What is a **Masking policy**?  

A: A policy that dynamically obfuscates data at query time based on roles or conditions, protecting sensitive fields. 

Q: What is a **Row access policy**?  

A: A predicate-based policy enforcing row-level security to control which records a user can see. 

---

## Data Engineering & Ingestion

Q: What is **COPY INTO**?  

A: The SQL command to bulk load data from a stage into a table (or unload from a table to a stage), with options for formats and validation. 

Q: What is **Snowpipe (auto-ingest)**?  

A: A serverless service that automatically loads new files as they land in a stage using event notifications for low-latency ingestion. 

Q: What is **Snowpipe Streaming**?  

A: A streaming ingestion interface for row-level inserts with low latency, ideal for event/IoT or micro-batch scenarios. 

Q: What are **Streams and Tasks** together?  

A: Streams provide CDC; Tasks schedule SQL—combined, they power incremental ELT pipelines and materializations. 

Q: What is **File staging (S3; Azure Blob; GCS)**?  

A: Placing data files in cloud storage referenced by external stages (or internal Snowflake stages) for loading/unloading. 

Q: What is **Semi-structured data (VARIANT; OBJECT; ARRAY)**?  

A: Native types for JSON-like data with flexible schemas and SQL functions for traversal and transformation. 

Q: What are **File formats (JSON; Parquet; Avro; ORC; XML)**?  

A: Supported structured and semi-structured formats for efficient load/unload and query. 

Q: What are **Geospatial data types (GEOGRAPHY; GEOMETRY)**?  

A: Types and functions for storing and analyzing geospatial features, enabling spatial joins and operations. 

Q: What is the **Search Optimization Service (SOS)**?  

A: A service that maintains auxiliary search access paths to accelerate selective queries without manual indexing. 

Q: What is the **Query Acceleration Service (QAS)**?  

A: A service that elastically adds compute resources to speed up large or complex queries without resizing warehouses. 

---

## Performance & Storage Optimization

Q: What are **Micro-partitions**?  

A: Snowflake's immutable storage blocks with rich metadata used for pruning and fast scan planning. 

Q: What is **Clustering** (clustering depth)?  

A: Organizing data to align with common filters/joins; clustering depth indicates how well data is clustered. 

Q: What is **Pruning**?  

A: Skipping micro-partitions that cannot match a query based on metadata, reducing I/O and improving performance. 

Q: What is the **Result cache**?  

A: Cached final query results returned instantly when the same query repeats with unchanged data. 

Q: What is the **Metadata cache**?  

A: Cached stats and metadata (e.g., micro-partition ranges) used by the optimizer to plan efficient scans. 

Q: What is **Warehouse sizing and scaling policy** (auto-suspend/resume; multi-cluster)?  

A: Controls for right‑sizing compute and handling concurrency spikes via auto‑suspend/resume and multi‑cluster scaling. 

Q: What are **Resource monitors**?  

A: Credit usage guards that alert or suspend warehouses when thresholds are reached to control spend. 

Q: What is **Time Travel**?  

A: The ability to query historical data versions and recover objects within a retention window. 

Q: What is **Fail-safe**?  

A: A post–Time Travel recovery period for disaster scenarios; intended for Snowflake-managed recovery. 

Q: What is **Zero-Copy Cloning**?  

A: Instant creation of full logical copies of databases/schemas/tables without duplicating stored data. 

---

## Governance & Security

Q: What is **RBAC (role-based access control)**?  

A: A permissions model where roles own objects and grant privileges to control access consistently. 

Q: What are **System roles** (ACCOUNTADMIN; SYSADMIN; SECURITYADMIN; USERADMIN; ORGADMIN)?  

A: Built-in high-privilege roles for account and org administration, security, and object management. 

Q: What are **Privileges and object ownership**?  

A: Ownership grants full control of an object; specific privileges (SELECT, USAGE, etc.) enable actions by other roles. 

Q: What is a **Network policy**?  

A: IP allow/deny configurations restricting where users/services can connect from. 

Q: What is **MFA/SSO (SAML; OIDC)**?  

A: Authentication integrations enabling multi-factor auth and single sign-on with identity providers via SAML/OIDC. 

Q: What is **OAuth (native; external)**?  

A: Token-based delegated access, either through Snowflake's native authorization server or external providers. 

Q: What is **Key pair authentication**?  

A: Certificate-based authentication for service users and automation (private/public key pairs). 

Q: What are **Customer-Managed Keys (CMK) / Tri-Secret Secure**?  

A: Encryption options where customers control keys and add an extra key component for enhanced protection. 

Q: What is **Tag-based masking**?  

A: Applying masking policies based on tags to consistently protect sensitive fields across objects. 

Q: What is **Data Classification**?  

A: Automated and manual techniques to detect, label, and govern sensitive data. 

---

## Data Sharing & Collaboration

Q: What is **Secure Data Sharing**?  

A: A way to share live data between accounts without copying, preserving a single source of truth. 

Q: Who are **Provider / Consumer**?  

A: The provider exposes data sets via shares or listings; the consumer reads them in their account. 

Q: What are **Reader accounts**?  

A: Lightweight Snowflake accounts for consumers who don't have a full Snowflake account. 

Q: What is **Snowflake Marketplace** (listings; private listings)?  

A: A catalog to publish and subscribe to datasets, functions, and apps; supports public and private listings. 

Q: What is **Database replication**?  

A: Copying databases across accounts/regions/clouds for DR, proximity, or read scaling. 

Q: What is **Failover / Failback**?  

A: Switching to a replica during outage and returning to primary afterward, maintaining continuity. 

Q: What is **Cross-region / Cross-cloud replication (Snowgrid)**?  

A: Platform features enabling multi‑region/multi‑cloud data and governance replication across Snowflake's grid. 

Q: What are **Data Clean Rooms**?  

A: Privacy-preserving collaboration spaces to analyze combined datasets with strict controls. 

Q: What is **Resharing**?  

A: Allowing downstream consumers to share data further, subject to provider policies. 

---

## AI/ML and Developer Ecosystem

Q: What is **Snowpark** (Python; Java; Scala)?  

A: A developer framework to write data pipelines and ML logic in familiar languages executed close to your data. 

Q: What are **Snowpark Container Services** (compute pools; services)?  

A: A managed container runtime to run services and workloads (e.g., Python apps, vector DBs) inside Snowflake compute pools. 

Q: What are **Snowflake Notebooks**?  

A: Integrated notebooks for data exploration, analytics, and ML inside the Snowflake UI. 

Q: What is **Snowflake AI & ML Studio**?  

A: An integrated experience to build, evaluate, and operationalize AI/ML workflows on Snowflake. 

Q: What is **Snowflake Cortex AI**?  

A: A platform capability that brings LLM-powered functions, agents, and retrieval to Snowflake data. 

Q: What are **Cortex functions** (COMPLETE; SUMMARIZE; TRANSLATE; SENTIMENT; CLASSIFY; EXTRACT_ANSWER)?  

A: Built‑in AI SQL functions that call hosted models for text generation, summarization, translation, sentiment, classification, and QA. 

Q: What is **Vector search / embeddings (EMBED_768; EMBED_1024)**?  

A: Functions to embed text into vectors and perform similarity search for retrieval‑augmented experiences. 

Q: What is **Snowflake Arctic**?  

A: An efficient, enterprise‑grade open LLM family designed for cost‑effective performance on core tasks. 

Q: What is **Open Catalog (based on Apache Polaris)**?  

A: An open interoperability layer for lakehouse catalogs built on the Polaris spec. 

Q: What is **Streamlit in Snowflake**?  

A: A serverless way to build and deploy Streamlit apps directly on data in Snowflake. 

Q: What is the **Native Apps Framework**?  

A: A framework to build, distribute, and monetize applications running in customer accounts via Marketplace. 

Q: What is **Snowflake CLI**?  

A: A command-line tool to interact with Snowflake services and developer workflows. 

Q: What is **SnowSQL**?  

A: Snowflake's SQL command-line client for interactive use and scripting. 

Q: What are **Connectors** (JDBC; ODBC; Python; Spark; Kafka; Node.js; .NET; Go)?  

A: Client drivers and integrations enabling data movement, querying, and orchestration from external systems. 

Q: What is the **SQL API**?  

A: A REST API to execute SQL statements and manage sessions programmatically. 

---

## Monitoring, Metadata, and Tooling

Q: What is **ACCOUNT_USAGE**?  

A: A shared database exposing detailed account telemetry (queries, access, costs) for observability. 

Q: What is **INFORMATION_SCHEMA**?  

A: System views that expose metadata about objects within each database/schema. 

Q: What is **Query history**?  

A: Logs and views detailing past queries, performance, and resource use. 

Q: What is **Access history**?  

A: Auditable records of who accessed what data and when. 

Q: What is **Resource usage**?  

A: Metrics and tables showing compute and storage consumption, typically credit-based. 

Q: What is a **Query profile**?  

A: An execution plan and performance breakdown visualization for a specific query. 

Q: What is **Snowsight** (worksheets; dashboards; notebooks; Streamlit apps)?  

A: Snowflake's UI for authoring SQL, building dashboards, running notebooks, and deploying Streamlit apps. 

---

## Internal Glossaries

Q: What is the **Balto Design System glossary**?  

A: An internal glossary for UI/UX design language and components used across Snowflake products. 

Q: What is the **Performance terms glossary**?  

A: An internal glossary defining performance-related concepts and metrics for Snowflake engineering. 

---

## Editions, Deployment, and Architecture

Q: What is the **Standard edition**?  

A: The base product tier with core Snowflake capabilities. 

Q: What is the **Enterprise edition**?  

A: Adds advanced features for performance, governance, and business needs beyond Standard. 

Q: What is the **Business Critical edition**?  

A: A higher-assurance tier with enhanced security and compliance controls. 

Q: What is **Virtual Private Snowflake (VPS)**?  

A: A deployment with increased isolation for stringent security/regulatory requirements. 

Q: What are **Gov/regulated deployments**?  

A: Snowflake regions and setups tailored for government and regulated industries. 

Q: What are **AWS regions**, **Azure regions**, **GCP regions**?  

A: Cloud-provider regions where Snowflake operates, chosen for proximity, compliance, and redundancy. 

Q: What is **Snowgrid**?  

A: Snowflake's cross-region and cross-cloud mesh enabling replication, failover, and governance at scale. 

Q: What is an **Account locator / account URL**?  

A: The unique identifier and URL used for connecting to a specific Snowflake account/region. 

Q: Who is the **Organization-level admin (ORGADMIN)**?  

A: The role that manages organization-wide settings, accounts, and governance. 

---

## Pricing & Usage

Q: What are **Credits**?  

A: The unit of Snowflake consumption for compute/services; cost depends on warehouse size and feature usage. 

Q: What is **On-demand vs capacity usage**?  

A: Pay-as-you-go credit consumption vs pre-purchased capacity commitments with potential discounts. 

Q: What is **Auto-suspend / Auto-resume**?  

A: Warehouse policies that pause compute when idle and restart on demand to control spend. 

Q: What are **Resource monitors** in billing context?  

A: Threshold-based controls that notify or suspend to cap credit consumption. 

Q: What is **Marketplace billing**?  

A: Charges associated with consuming or distributing data/app listings via Marketplace. 

---

## New AI Product Terminology

### Brand Platforms

Q: What is **Snowflake Cortex AI**?  

A: A platform layer delivering LLM-based functions, agents, and retrieval integrated with Snowflake data and governance. 

Q: What is **Snowflake AI & ML Studio**?  

A: A unified workspace to build, evaluate, and manage AI/ML workflows on Snowflake. 

Q: What is **Snowflake Intelligence**?  

A: Umbrella positioning for AI-powered experiences and copilots across the Snowflake product surface. 

Q: What is **Snowflake Copilot**?  

A: An AI assistant embedded in Snowflake experiences to accelerate tasks like SQL authoring and analysis. 

### Core AI Capabilities

Q: What is **Cortex Analyst**?  

A: An AI assistant that answers questions over your data using governed retrieval and reasoning. 

Q: What are **Cortex Agents**?  

A: Configurable, tool-using agents that can plan, call functions, and act within governed Snowflake contexts. 

Q: What is **Cortex AISQL**?  

A: AI-assisted SQL generation and transformation capabilities surfaced in the product and via functions. 

Q: What is **Cortex Search**?  

A: A semantic search and retrieval service over your enterprise data with vector capabilities. 

Q: What are **Cortex Knowledge Extensions**?  

A: Connectors and retrieval adapters that let AI leverage enterprise knowledge sources. 

Q: What is **Cortex Fine-Tuning**?  

A: Tools to adapt foundation models to your domain using governed data. 

Q: What is **Cortex Guard**?  

A: Safety, compliance, and observability controls for AI functions and applications. 

### AI SQL Functions

Q: What does **COMPLETE** do?  

A: Generates text (LLM completion) from prompts/inputs in SQL. 

Q: What does **SUMMARIZE** do?  

A: Produces concise summaries of text inputs. 

Q: What does **TRANSLATE** do?  

A: Translates text between languages. 

Q: What does **SENTIMENT** do?  

A: Classifies tone/polarity of text (e.g., positive/negative). 

Q: What does **CLASSIFY** do?  

A: Assigns categories/labels to text based on instructions or examples. 

Q: What does **EXTRACT_ANSWER** do?  

A: Answers a question using supplied context passages. 

Q: What does **EXTRACT** do?  

A: Pulls structured fields or entities from unstructured text. 

Q: What does **PARSE_DOCUMENT** do?  

A: Parses documents (e.g., PDFs) to extract text and structured elements. 

Q: What does **TRANSCRIBE** do?  

A: Converts speech/audio into text. 

Q: What does **EMBED / EMBED_TEXT_768 / EMBED_TEXT_1024** do?  

A: Creates vector embeddings of text for similarity search and RAG. 

Q: What does **AI_AGG / SUMMARIZE_AGG** do?  

A: Aggregates groups of rows using AI (e.g., summarizing many records per group). 

Q: What are **AI_FILTER (preview/roadmap)** and **AI_SIMILARITY (preview/roadmap)**?  

A: Functions for AI-based filtering and similarity operations planned/previewed in SQL. 

Q: What is **AI_CLUSTER (preview/roadmap)**?  

A: An AI-driven clustering function to group similar records without manual features. 

### Notebooks Runtime

Q: What are **Snowflake Notebooks**, **Container Runtime**, **Warehouse Runtime**?  

A: Integrated notebooks leveraging either container-based execution or warehouse compute, depending on workload needs. 

### Governance & Quality

Q: What is **AI observability**?  

A: Monitoring AI pipelines for quality, drift, cost, and safety, with traceability and controls. 

---

## Snova / Cortex Code Terminology

### Names & Positioning

Q: What is **Snova**?  

A: An internal codename historically used for AI/agent tooling that evolved into Cortex Code experiences. 

Q: What is the **Cortex Code CLI (formerly Snova)**?  

A: A command-line interface for developing and running Cortex-powered workflows from your terminal. 

Q: What is **Cortex Code in Snowsight**?  

A: The UI-integrated experience to run and manage Cortex Code interactions directly in Snowsight. 

### Accounts, Connections, Architecture

Q: What is an **Inference account** (e.g., Snowhouse)?  

A: A Snowflake-managed account that hosts AI inference services used by functions and agents. 

Q: What is an **Active account**?  

A: The user's current Snowflake account context where commands and workloads execute. 

Q: What are **PAT-based connections** (config in ~/.snowflake/config.toml or connections.toml)?  

A: Personal Access Token configurations for authenticating CLI tools to Snowflake. 

Q: What does `cortex -c <connection_name>` mean?  

A: A CLI flag to select a named connection profile for running Cortex Code commands. 

### Agent Features, Modes, and Controls

Q: What is **Planning mode** (`/plan`; `/plan_off`)?  

A: A mode where the agent shows proposed step-by-step plans before execution for transparency and control. 

Q: What is **Review mode** (`/review`)?  

A: A mode prompting the agent to explain or validate actions and outputs for oversight. 

Q: What are **Slash commands** (`/sql`; `/sh`; `/feedback`)?  

A: Commands that direct the agent to run SQL, shell operations, or collect feedback inline. 

Q: What are **Skills**?  

A: Tool or capability plugins the agent can call (SQL, HTTP, python, etc.) under governance. 

Q: What is the **Model Context Protocol (MCP)**?  

A: A protocol for connecting tools and contexts to LLMs in a standardized way. 

Q: What is **Session continuity** (`cortex --continue; --resume`)?  

A: Flags to continue or resume previous agent sessions and context. 

Q: What are **Permissions controls** (`--dangerously-allow-all-tool-calls`)?  

A: A CLI safety override that allows the agent to call any tool; use cautiously. 

### Status & Enablement

Q: What is **Private Preview (PrPr)**?  

A: A limited-access program for early testing of new features prior to public preview/GA. 

Q: What are **Internal links** like `go/cortex-cli` or `go/snova`?  

A: Company-shortened URLs that redirect to internal documentation, tooling, or dashboards. 

---

## Quick Reference: Common Distinctions

- **Permanent vs Transient vs Temporary tables:** Retention and scope tradeoffs (full retention vs reduced vs session-only). 

- **Standard vs Secure views:** Security posture; secure views hide definitions for safer sharing. 

- **Materialized view vs Dynamic table:** Both maintain results; dynamic tables emphasize incremental, declarative pipelines. 

- **Internal vs External stages:** Storage location (in Snowflake vs in your cloud storage). 

- **SOS vs QAS:** SOS adds search access paths; QAS elastically adds compute to speed queries. 

- **Time Travel vs Fail-safe:** User-accessible history window vs Snowflake-managed recovery period. 

- **Share vs Listing:** Direct account-to-account share vs Marketplace distribution. 

