This dissertation focuses on the design and implementation of a prototype AI-assisted SRE framework that supports root cause analysis of application failures using operational log data. The system will operate in a controlled cloud-native environment deployed locally using a container orchestration platform.

The work includes developing a mechanism to automatically collect logs from multiple microservices running in a local containerized cluster, preprocess and structure the collected data, and provide it to an AI-based analysis component. The system will use Large Language Models (LLMs) to interpret log data and generate structured diagnostic explanations that assist engineers in understanding system failures.

The framework will incorporate contextual system information, such as service relationships and runtime metadata, to improve the relevance and interpretability of the AI-generated analysis. Interaction with the system will be provided through a command-line interface, allowing engineers to query failures and receive diagnostic insights.

The implementation will be demonstrated using a locally deployed microservices application running in a Minikube-based Kubernetes environment, where controlled failure scenarios will be generated to evaluate system behaviour.

The scope of the project is limited to prototype development and evaluation in a local environment. It does not include deployment in large-scale production infrastructure, real-time monitoring integration with enterprise observability platforms, or automated remediation of system failures. The focus is on assisting analysis and diagnosis rather than performing autonomous system recovery.

The primary deliverables of this work include the system architecture design, implementation of the AI-assisted analysis pipeline, integration of log collection and contextual data handling, and evaluation through representative incident scenarios.