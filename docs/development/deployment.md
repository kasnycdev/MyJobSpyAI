# Deployment Guide

## Overview

This guide provides comprehensive information about deploying MyJobSpyAI in various environments, including local development, cloud platforms, and container orchestration.

## Deployment Options

```mermaid
graph TD
    A[Deployment Options] --> B[Local Development]
    A --> C[Cloud Platforms]
    A --> D[Container Orchestration]
    B --> E[Docker]
    B --> F[Virtualenv]
    C --> G[AWS]
    C --> H[Google Cloud]
    C --> I[Azure]
    D --> J[Kubernetes]
    D --> K[Docker Swarm]
```

## Local Development

### Docker Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  myjobspyai:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
    volumes:
      - .:/app
    command: python -m myjobspyai
```

### Virtual Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python -m myjobspyai
```

## Cloud Platforms

### AWS Deployment

```yaml
# AWS CloudFormation Template
AWSTemplateFormatVersion: '2010-09-09'
Description: 'MyJobSpyAI Deployment'

Resources:
  EC2Instance:
    Type: 'AWS::EC2::Instance'
    Properties:
      ImageId: ami-0c55b159cbfafe1f0
      InstanceType: t3.medium
      SecurityGroupIds:
        - !Ref SecurityGroup
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          yum update -y
          yum install -y docker
          service docker start
          usermod -a -G docker ec2-user
          curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
          chmod +x /usr/local/bin/docker-compose
          git clone https://github.com/yourusername/myjobspyai.git
          cd myjobspyai
          docker-compose up -d

  SecurityGroup:
    Type: 'AWS::EC2::SecurityGroup'
    Properties:
      GroupDescription: 'Allow HTTP and HTTPS'
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myjobspyai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myjobspyai
  template:
    metadata:
      labels:
        app: myjobspyai
    spec:
      containers:
      - name: myjobspyai
        image: yourusername/myjobspyai:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"

# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: myjobspyai
spec:
  selector:
    app: myjobspyai
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Container Orchestration

### Docker Swarm

```yaml
# docker-stack.yml
version: '3.8'

services:
  myjobspyai:
    image: yourusername/myjobspyai:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.50'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
    ports:
      - "8000:8000"
    networks:
      - myjobspyai-network

networks:
  myjobspyai-network:
    driver: overlay
```

## Environment Configuration

```yaml
# .env.example
DATABASE_URL=postgresql://user:password@localhost:5432/myjobspyai
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key-here
DEBUG=True
```

## Monitoring and Logging

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'myjobspyai'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Logging Configuration

```yaml
# logging.yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    formatter: default
    level: DEBUG

loggers:
  myjobspyai:
    level: DEBUG
    handlers: [console]
    propagate: False
```

## Security Considerations

!!! warning "Security Best Practices"
    - Use environment variables for sensitive data
    - Enable HTTPS in production
    - Use proper authentication
    - Regular security audits
    - Keep dependencies up to date
    - Use security headers
    - Implement rate limiting
    - Use proper CORS configuration

## Troubleshooting

### Common Issues

=== "Docker"
    ```bash
    # Docker build issues
    docker build --no-cache .
    
    # Container restarts
    docker logs container_id
    
    # Port conflicts
    docker ps -a
    docker stop container_id
    ```

=== "Kubernetes"
    ```bash
    # Pod status
    kubectl get pods
    
    # Pod logs
    kubectl logs pod_name
    
    # Resource usage
    kubectl top pods
    ```

=== "AWS"
    ```bash
    # EC2 status
    aws ec2 describe-instances
    
    # CloudWatch logs
    aws logs get-log-events --log-group-name /aws/lambda/myjobspyai
    
    # Security groups
    aws ec2 describe-security-groups
    ```

## Best Practices

!!! tip "Deployment Best Practices"
    - Use CI/CD pipelines
    - Implement proper logging
    - Monitor performance
    - Use environment variables
    - Implement proper error handling
    - Use proper resource limits
    - Regular backups
    - Security updates
    - Regular testing
