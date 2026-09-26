# Azure deployment map

- Run the container in Azure Container Apps with min/max replicas and managed identity.
- Store API/provider secrets in Key Vault and reference them from Container Apps.
- Replace local seams with Azure Cache for Redis and Azure Database for PostgreSQL Flexible Server.
- Put the gateway behind API Management, require TLS, and export Prometheus/OpenTelemetry data to Azure Monitor.
- Run migrations as a one-off deployment job before shifting traffic.
