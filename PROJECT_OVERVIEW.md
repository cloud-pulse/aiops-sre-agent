**complete project definition document** that explains:

✔ What the project is
✔ Why it exists
✔ What problem it solves
✔ How it works internally
✔ System architecture
✔ Execution flow
✔ What makes it different from a chatbot
✔ What components exist
✔ How input is handled
✔ What output is produced
✔ Implementation boundaries
✔ Research contribution

This will act as your **single source of truth**.
If memory is lost, you paste this → I immediately understand everything.


---

# 📄 ARTIFACT 1 — FULL PROJECT DEFINITION

*(Save as PROJECT_OVERVIEW.md)*

---

# AI-Assisted Root Cause Analysis for Cloud-Native Microservices

## Local Dependency-Aware Log Investigation System

---

## 1. Project Summary

This project implements a **local AI-assisted Site Reliability Engineering (SRE) debugging system** that performs **dependency-aware root cause analysis** for failures in distributed microservice environments.

The system runs microservices locally (Minikube), collects logs from multiple interdependent services, correlates failure events using a service dependency graph, identifies the most probable root cause, and generates a human-readable diagnostic explanation using a Large Language Model (LLM).

Unlike traditional AI log analysis tools that interpret isolated log snippets, this system performs **multi-service failure correlation and propagation tracing**, enabling it to identify the actual origin of failures rather than just describing symptoms.

---

## 2. Problem Statement

Modern cloud-native systems are composed of multiple interdependent microservices. Failures rarely occur in isolation — they propagate through service dependency chains.

Example:

```
Payment service fails
→ API cannot connect
→ BFF fails
→ UI shows 500 error
```

However, engineers typically observe only the **symptom**, not the cause.

Existing debugging approaches involve:

* manually inspecting logs across services
* correlating timestamps
* tracing dependencies
* identifying origin of failure

This process is time-consuming, cognitively demanding, and error-prone.

AI tools that simply interpret logs are insufficient because they lack:

* service dependency awareness
* cross-service correlation
* failure propagation reasoning
* system topology context

Therefore, a context-aware, system-level debugging assistant is required.

---

## 3. Core Objective

To design and implement a system that:

1. Understands relationships between microservices
2. Collects logs from multiple related services
3. Correlates failure events across services
4. Identifies the true origin of failure
5. Explains failure propagation to engineers
6. Provides diagnostic reasoning using AI

---

## 4. What the System Is (Conceptually)

The system is **not a chatbot**.

It is a:

### Dependency-Aware Failure Investigation Engine

*

### AI Explanation Generator

The system performs structured reasoning first, then uses AI to explain results.

---

## 5. System Environment

* Local Kubernetes cluster using Minikube
* Multiple interdependent microservices
* Failures intentionally triggered or naturally occurring
* CLI-based investigation tool

No cloud deployment required.

---

## 6. Core System Knowledge

The system maintains three knowledge layers:

### 6.1 Service Topology (Persistent System Knowledge)

Static dependency graph describing service relationships.

Example:

```
UI → BFF → API → Payment → Database
```

Loaded once at system startup.

---

### 6.2 Runtime Observability Data

Logs collected from services during investigation.

---

### 6.3 Historical Incident Knowledge (Optional)

Previously observed failures stored in vector database for contextual retrieval (RAG).

---

## 7. High-Level Architecture

```
Microservices (Minikube)
        ↓
Log Collection
        ↓
Log Preprocessing
        ↓
Dependency Graph Reasoning
        ↓
Multi-Service Failure Correlation
        ↓
Root Cause Candidate Detection
        ↓
Context Retrieval (optional RAG)
        ↓
LLM Explanation
        ↓
CLI Output to SRE
```

---

## 8. How the System Works (Operational Flow)

### Step 1 — Failure Trigger

User reports a failing service or endpoint.

---

### Step 2 — Dependency Awareness

System loads service relationship graph.

---

### Step 3 — Investigation Scope Determination

System identifies upstream and downstream services.

---

### Step 4 — Multi-Service Log Collection

Logs collected from all relevant services.

---

### Step 5 — Log Preprocessing

Logs cleaned, structured, and timestamped.

---

### Step 6 — Temporal Correlation

System reconstructs event timeline.

---

### Step 7 — Failure Propagation Analysis

System identifies which service failed first.

---

### Step 8 — Root Cause Candidate Detection

Dependency + timeline reasoning used to detect origin.

---

### Step 9 — Context Augmentation (Optional)

Similar incidents retrieved from vector database.

---

### Step 10 — AI Explanation

LLM receives structured incident summary and produces human-readable diagnosis.

---

### Step 11 — Diagnostic Output

CLI presents root cause and recommended actions.

---

## 9. Example Investigation

Observed symptom:

```
API returning 500 errors
```

Detected sequence:

```
10:00 Payment service crash
10:01 API connection failure
10:02 BFF failure
10:03 UI error
```

Root cause:

```
Payment service failure
```

AI explanation describes propagation chain.

---

## 10. System Inputs

The system supports:

1. Failing service name
2. Pod name
3. Namespace
4. Log file
5. Pasted logs

---

## 11. System Outputs

The system produces:

* root cause identification
* failure propagation chain
* impacted services
* diagnostic explanation
* suggested remediation

---

## 12. User Interaction Model

CLI-based investigation tool.

Example:

```
investigate api-service
```

---

## 13. What Makes This System Different From AI Chatbots

Typical AI log tools:

* interpret text
* lack system awareness

This system:

* understands service topology
* correlates multi-service events
* reconstructs failure timeline
* identifies origin of failure

AI is used only for explanation, not detection.

---

## 14. Research Contribution

This project demonstrates that:

Dependency-aware multi-service log correlation combined with contextual AI reasoning improves root cause analysis in distributed microservice systems.

---

## 15. Implementation Scope

Included:

* local microservice environment
* dependency graph modeling
* automated multi-service log collection
* failure propagation tracing
* AI explanation generation

Not included:

* cloud deployment
* automated remediation
* monitoring dashboards
* CI/CD integration

---

## 16. Conceptual System Model

The system behaves like:

A forensic investigator reconstructing a system failure timeline, then an AI assistant explaining the findings.

---

## 17. Final System Definition

A context-aware AI-assisted root cause analysis tool that performs dependency-driven multi-service log correlation to identify and explain failure origins in distributed microservice environments.

---

# END OF ARTIFACT

---
