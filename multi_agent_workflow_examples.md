# Multi-Agent示例工作流配置

## 1. 代码审查工作流

```yaml
# config/workflows/code_review.yaml
name: code_review
version: "2.0"
description: "自动化代码审查工作流 - 使用Sequential Pipeline模式"

workflow:
  pattern: sequential_pipeline
  allow_feedback: true
  max_iterations: 2
  timeout: 300  # 5分钟超时

agents:
  # 执行者Agent
  - id: syntax_analyzer
    role: executor
    specialization: syntax_analysis
    config:
      max_concurrent_tasks: 3
      timeout_seconds: 60
    personality:
      personality: 80
      motivations: 85
      growth: 75
    capabilities:
      - syntax_check
      - style_check
      - linting

  - id: security_scanner
    role: executor
    specialization: security_analysis
    config:
      max_concurrent_tasks: 2
      timeout_seconds: 120
    personality:
      personality: 75
      conflict: 85
      emotions: 70
    capabilities:
      - vulnerability_scan
      - dependency_check
      - secret_detection

  - id: performance_analyzer
    role: executor
    specialization: performance_analysis
    config:
      max_concurrent_tasks: 2
      timeout_seconds: 90
    personality:
      personality: 80
      motivations: 80
    capabilities:
      - complexity_analysis
      - performance_profiling
      - bottleneck_detection

  # 验证者Agent
  - id: quality_validator
    role: validator
    validation_types:
      - syntax
      - logic
      - security
    config:
      max_concurrent_tasks: 5
    personality:
      personality: 75
      conflict: 80
      emotions: 70

  # 批评者Agent
  - id: code_critic
    role: critic
    config:
      max_concurrent_tasks: 2
    personality:
      personality: 70
      conflict: 90
      emotions: 65
    capabilities:
      - code_review
      - best_practices_check
      - maintainability_analysis

execution:
  steps:
    # 第1步: 语法分析
    - name: syntax_check
      agent: syntax_analyzer
      action: analyze_syntax
      input: "{{input.code}}"
      output: syntax_report
      
    # 第2步: 安全扫描
    - name: security_scan
      agent: security_scanner
      action: scan_security
      input: "{{input.code}}"
      output: security_report
      
    # 第3步: 性能分析
    - name: performance_analysis
      agent: performance_analyzer
      action: analyze_performance
      input: "{{input.code}}"
      output: performance_report
      
    # 第4步: 质量批评
    - name: quality_review
      agent: code_critic
      action: review_code
      input:
        code: "{{input.code}}"
        syntax_report: "{{syntax_report}}"
        security_report: "{{security_report}}"
        performance_report: "{{performance_report}}"
      output: review_feedback
      
    # 第5步: 验证结果
    - name: validate
      agent: quality_validator
      action: validate_quality
      input:
        code: "{{input.code}}"
        feedback: "{{review_feedback}}"
      output: validation_result
      
    # 反馈循环条件
    - condition: validation_result.quality_score < 80
      action: feedback_loop
      max_iterations: 2
      notify: true

acceptance_criteria:
  - metric: syntax_score
    operator: ">="
    value: 95
  - metric: security_score
    operator: ">="
    value: 90
  - metric: performance_score
    operator: ">="
    value: 80
  - metric: overall_quality
    operator: ">="
    value: 85

notifications:
  on_success:
    - type: webhook
      url: "{{config.webhook_url}}"
      payload:
        status: "passed"
        review_id: "{{input.review_id}}"
        
  on_failure:
    - type: webhook
      url: "{{config.webhook_url}}"
      payload:
        status: "failed"
        review_id: "{{input.review_id}}"
        errors: "{{validation_result.errors}}"
        
  on_feedback:
    - type: comment
      platform: github
      pr_number: "{{input.pr_number}}"
      message: "{{review_feedback}}"

logging:
  level: INFO
  destination:
    - type: file
      path: "/var/log/multi-agent/code_review.log"
    - type: elasticsearch
      index: "code-review-logs"
```

## 2. 研究任务工作流

```yaml
# config/workflows/research_task.yaml
name: research_task
version: "2.0"
description: "多Agent协作研究任务 - 使用Hierarchical模式"

workflow:
  pattern: hierarchical
  timeout: 1800  # 30分钟超时

agents:
  # 战略层
  orchestrator:
    id: research_lead
    role: orchestrator
    config:
      max_concurrent_tasks: 10
      timeout_seconds: 1800
    personality:
      personality: 90
      motivations: 95
      conflict: 85
      relationships: 85
    capabilities:
      - goal_decomposition
      - strategy_planning
      - resource_scheduling
      - result_aggregation

  # 战术层
  coordinators:
    - id: data_collection_coord
      role: coordinator
      specialization: data_collection
      config:
        max_concurrent_tasks: 5
      personality:
        personality: 85
        relationships: 90
        emotions: 80
      capabilities:
        - task_orchestration
        - load_balancing
        - progress_monitoring

    - id: analysis_coord
      role: coordinator
      specialization: data_analysis
      config:
        max_concurrent_tasks: 5
      personality:
        personality: 85
        relationships: 90
        emotions: 80
      capabilities:
        - task_orchestration
        - quality_control
        - result_integration

    - id: synthesis_coord
      role: coordinator
      specialization: report_synthesis
      config:
        max_concurrent_tasks: 3
      personality:
        personality: 85
        relationships: 90
        emotions: 80
      capabilities:
        - content_integration
        - quality_assurance

  # 执行层
  executors:
    # 数据收集组
    - id: web_searcher
      role: executor
      specialization: web_search
      config:
        max_concurrent_tasks: 3
      capabilities:
        - web_search
        - content_extraction
        - source_validation

    - id: document_reader
      role: executor
      specialization: document_analysis
      config:
        max_concurrent_tasks: 3
      capabilities:
        - pdf_parsing
        - ocr_extraction
        - structured_data_extraction

    - id: database_query
      role: executor
      specialization: database_query
      config:
        max_concurrent_tasks: 2
      capabilities:
        - sql_query
        - nosql_query
        - data_retrieval

    # 数据分析组
    - id: data_processor
      role: executor
      specialization: data_processing
      config:
        max_concurrent_tasks: 3
      capabilities:
        - data_cleaning
        - data_transformation
        - statistical_analysis

    - id: trend_analyzer
      role: executor
      specialization: trend_analysis
      config:
        max_concurrent_tasks: 2
      capabilities:
        - time_series_analysis
        - pattern_recognition
        - trend_forecasting

    - id: insight_generator
      role: executor
      specialization: insight_generation
      config:
        max_concurrent_tasks: 2
      capabilities:
        - insight_extraction
        - hypothesis_generation
        - correlation_analysis

    # 报告生成组
    - id: report_writer
      role: executor
      specialization: report_writing
      config:
        max_concurrent_tasks: 2
      capabilities:
        - content_synthesis
        - report_formatting
        - citation_management

    - id: visual_designer
      role: executor
      specialization: data_visualization
      config:
        max_concurrent_tasks: 2
      capabilities:
        - chart_generation
        - infographic_design
        - dashboard_creation

  # 验证者
  validators:
    - id: fact_checker
      role: validator
      validation_types:
        - fact
        - source
      capabilities:
        - fact_verification
        - source_credibility_check

    - id: quality_assurer
      role: validator
      validation_types:
        - completeness
        - coherence
      capabilities:
        - content_completeness_check
        - logical_coherence_check

execution:
  decomposition:
    strategy: parallel_subtasks
    phases:
      - name: data_collection_phase
        coordinator: data_collection_coord
        executors: [web_searcher, document_reader, database_query]
        parallel: true
        timeout: 600
        
      - name: data_analysis_phase
        coordinator: analysis_coord
        executors: [data_processor, trend_analyzer, insight_generator]
        depends_on: [data_collection_phase]
        parallel: true
        timeout: 600
        
      - name: synthesis_phase
        coordinator: synthesis_coord
        executors: [report_writer, visual_designer]
        depends_on: [data_analysis_phase]
        parallel: false
        timeout: 300

  validation:
    checkpoints:
      - after: data_collection_phase
        validators: [fact_checker]
        criteria:
          - source_count >= 10
          - fact_accuracy >= 0.9
          
      - after: data_analysis_phase
        validators: [quality_assurer]
        criteria:
          - insight_count >= 5
          - data_coverage >= 0.8
          
      - after: synthesis_phase
        validators: [fact_checker, quality_assurer]
        criteria:
          - report_completeness >= 0.9
          - citation_accuracy >= 0.95

output:
  format: markdown
  sections:
    - executive_summary
    - methodology
    - findings
    - insights
    - recommendations
    - references
    - appendices

  artifacts:
    - type: document
      format: pdf
      name: "{{input.topic}}_research_report.pdf"
    - type: presentation
      format: pptx
      name: "{{input.topic}}_presentation.pptx"
    - type: data
      format: csv
      name: "{{input.topic}}_raw_data.csv"
```

## 3. 紧急事件响应工作流

```yaml
# config/workflows/incident_response.yaml
name: incident_response
version: "2.0"
description: "紧急事件响应工作流 - 使用Supervisor模式"

workflow:
  pattern: supervisor
  timeout: 300  # 5分钟超时
  check_interval: 5  # 5秒检查一次

priority: CRITICAL

agents:
  # 监督者
  supervisor:
    id: incident_commander
    role: supervisor
    config:
      max_concurrent_tasks: 10
      timeout_seconds: 300
    personality:
      personality: 90
      emotions: 75  # 冷静
      conflict: 85
    capabilities:
      - incident_assessment
      - resource_coordination
      - escalation_decision
      - stakeholder_communication

  # 执行者
  workers:
    - id: log_analyzer
      role: executor
      specialization: log_analysis
      config:
        max_concurrent_tasks: 2
        timeout_seconds: 60
      capabilities:
        - log_parsing
        - error_pattern_detection
        - root_cause_analysis

    - id: system_checker
      role: executor
      specialization: system_diagnostics
      config:
        max_concurrent_tasks: 3
        timeout_seconds: 60
      capabilities:
        - health_check
        - resource_monitoring
        - dependency_check

    - id: network_diagnostician
      role: executor
      specialization: network_diagnostics
      config:
        max_concurrent_tasks: 2
        timeout_seconds: 60
      capabilities:
        - connectivity_check
        - latency_analysis
        - traffic_analysis

    - id: security_investigator
      role: executor
      specialization: security_investigation
      config:
        max_concurrent_tasks: 2
        timeout_seconds: 90
      capabilities:
        - threat_detection
        - attack_vector_analysis
        - forensics_collection

    - id: auto_remediator
      role: executor
      specialization: automated_remediation
      config:
        max_concurrent_tasks: 2
        timeout_seconds: 120
      capabilities:
        - service_restart
        - configuration_rollback
        - traffic_rerouting

    - id: comm_agent
      role: executor
      specialization: communication
      config:
        max_concurrent_tasks: 5
        timeout_seconds: 30
      capabilities:
        - notification_dispatch
        - status_update
        - stakeholder_alert

execution:
  phases:
    # 第1阶段: 并行诊断
    - name: diagnosis
      parallel: true
      tasks:
        - agent: log_analyzer
          action: analyze_logs
          input:
            time_range: "{{incident.time_range}}"
            severity: "{{incident.severity}}"
          timeout: 60
          
        - agent: system_checker
          action: check_system_health
          input:
            services: "{{incident.affected_services}}"
          timeout: 60
          
        - agent: network_diagnostician
          action: diagnose_network
          input:
            endpoints: "{{incident.affected_endpoints}}"
          timeout: 60
          
        - agent: security_investigator
          action: investigate_security
          input:
            indicators: "{{incident.security_indicators}}"
          timeout: 90
          condition: incident.type == 'security'

    # 第2阶段: 评估与决策
    - name: assessment
      agent: incident_commander
      action: assess_impact
      input:
        diagnosis_results: "{{diagnosis.results}}"
      output: assessment_report

    # 第3阶段: 并行响应
    - name: response
      parallel: true
      tasks:
        - agent: auto_remediator
          action: apply_fixes
          input:
            assessment: "{{assessment_report}}"
            approved_actions: "{{assessment_report.recommended_actions}}"
          condition: assessment_report.auto_remediation_enabled
          timeout: 120
          
        - agent: comm_agent
          action: notify_stakeholders
          input:
            incident_id: "{{incident.id}}"
            severity: "{{assessment_report.severity}}"
            status: "responding"
          timeout: 30

  escalation:
    rules:
      - condition: assessment_report.severity == 'CRITICAL'
        action: page_oncall
        target: "{{config.oncall_engineer}}"
        
      - condition: response.duration > 180
        action: escalate_to_manager
        target: "{{config.engineering_manager}}"
        
      - condition: auto_remediator.failed
        action: require_manual_intervention
        notify: ["{{config.sre_team}}", "{{config.oncall_engineer}}"]
        
      - condition: security_investigator.threat_confirmed
        action: activate_security_protocol
        notify: ["{{config.security_team}}", "{{config.ciso}}"]

  recovery:
    verification:
      - agent: system_checker
        action: verify_recovery
        checks:
          - service_health
          - error_rate
          - response_time
          
      - agent: log_analyzer
        action: confirm_resolution
        criteria:
          - error_count == 0
          - normal_traffic_pattern

  post_incident:
    - agent: comm_agent
      action: send_resolution_notice
      
    - agent: log_analyzer
      action: generate_incident_report
      output: post_mortem_report
      
    - agent: incident_commander
      action: schedule_post_mortem
      attendees: "{{config.incident_response_team}}"

notifications:
  channels:
    - type: pagerduty
      service_key: "{{secrets.pagerduty_key}}"
      severity_mapping:
        CRITICAL: critical
        HIGH: error
        MEDIUM: warning
        LOW: info
        
    - type: slack
      webhook: "{{secrets.slack_webhook}}"
      channels:
        - "#incidents"
        - "#engineering-alerts"
        
    - type: email
      smtp_config: "{{config.smtp}}"
      recipients:
        - "{{config.oncall_email}}"
        - "{{config.sre_email}}"

logging:
  level: DEBUG
  include:
    - all_agent_communications
    - decision_points
    - action_timeline
    - state_changes
  destination:
    - type: file
      path: "/var/log/multi-agent/incidents/{{incident.id}}.log"
    - type: elasticsearch
      index: "incident-logs-{{incident.date}}"
```

## 4. 民主决策工作流

```yaml
# config/workflows/democratic_decision.yaml
name: democratic_decision
version: "2.0"
description: "多Agent民主协商决策 - 使用Democratic模式"

workflow:
  pattern: democratic
  consensus_threshold: 0.67  # 2/3多数
  max_rounds: 5
  timeout: 600  # 10分钟超时

agents:
  # 提案者
  proposers:
    - id: strategist_a
      role: proposer
      specialization: conservative_strategy
      config:
        max_concurrent_tasks: 2
      personality:
        personality: 85
        motivations: 90
        conflict: 70
      capabilities:
        - risk_assessment
        - conservative_planning
        - stability_analysis

    - id: strategist_b
      role: proposer
      specialization: aggressive_strategy
      config:
        max_concurrent_tasks: 2
      personality:
        personality: 85
        motivations: 90
        conflict: 80
      capabilities:
        - growth_optimization
        - market_opportunity_analysis
        - competitive_strategy

    - id: strategist_c
      role: proposer
      specialization: balanced_strategy
      config:
        max_concurrent_tasks: 2
      personality:
        personality: 85
        motivations: 90
        conflict: 75
      capabilities:
        - balanced_planning
        - stakeholder_analysis
        - sustainable_growth

  # 投票者
  voters:
    - id: analyst_1
      role: voter
      weight: 1.0
      specialization: financial_analysis
      capabilities:
        - financial_modeling
        - roi_calculation
        - risk_quantification

    - id: analyst_2
      role: voter
      weight: 1.0
      specialization: technical_analysis
      capabilities:
        - technical_feasibility
        - implementation_assessment
        - resource_estimation

    - id: analyst_3
      role: voter
      weight: 1.0
      specialization: market_analysis
      capabilities:
        - market_research
        - competitive_analysis
        - customer_impact_assessment

    - id: senior_analyst
      role: voter
      weight: 1.5  # 资深分析师权重更高
      specialization: strategic_analysis
      capabilities:
        - strategic_alignment
        - long_term_impact
        - portfolio_analysis

  # 共识验证者
  validators:
    - id: consensus_validator
      role: validator
      validation_types:
        - consensus_integrity
        - decision_quality
      capabilities:
        - vote_tally_verification
        - decision_rationality_check

execution:
  phases:
    # 第1阶段: 提案生成
    - name: proposal_generation
      agents: proposers
      action: generate_proposals
      input:
        decision_context: "{{input.decision_context}}"
        constraints: "{{input.constraints}}"
        objectives: "{{input.objectives}}"
      output: proposals
      timeout: 120

    # 第2阶段: 方案展示与讨论
    - name: deliberation
      agents: all
      action: discuss_proposals
      input:
        proposals: "{{proposals}}"
      rounds: 3
      timeout: 180

    # 第3阶段: 投票
    - name: voting
      agents: voters
      action: vote
      method: approval_voting  # 认可投票制
      input:
        proposals: "{{proposals}}"
        discussion_summary: "{{deliberation.summary}}"
      output: votes
      timeout: 60

    # 第4阶段: 共识检查
    - name: consensus_check
      agent: consensus_validator
      action: validate_consensus
      input:
        votes: "{{votes}}"
        threshold: "{{workflow.consensus_threshold}}"
      output: consensus_result

    # 第5阶段: 结果处理
    - name: result_resolution
      condition: consensus_result.reached
      then:
        action: finalize_decision
        output: final_decision
      else:
        action: handle_no_consensus
        options:
          - escalate_to_human
          - leader_override
          - extend_discussion

voting:
  method: approval_voting
  rules:
    - each_voter_can_approve_multiple_proposals: true
    - abstention_allowed: true
    - reasoning_required: true
    
  tie_breaking:
    method: weighted_random
    weights: voter_weights

fallback:
  no_consensus:
    - after_round: 3
      action: leader_proposal
      leader: senior_analyst
      
    - after_round: 5
      action: escalate_to_human
      notify: "{{input.decision_owner}}"

output:
  decision_record:
    - winning_proposal
    - vote_distribution
    - consensus_rate
    - dissenter_opinions
    - decision_rationale
    
  artifacts:
    - type: document
      name: "decision_record_{{input.decision_id}}.md"
    - type: data
      name: "vote_data_{{input.decision_id}}.json"

audit:
  record:
    - all_proposals
    - all_votes_with_reasoning
    - discussion_transcript
    - consensus_calculation
    
  retention: 7_years
```

## 5. 内容生成工作流

```yaml
# config/workflows/content_generation.yaml
name: content_generation
version: "2.0"
description: "高质量内容生成工作流 - 使用Critic-Reviewer模式"

workflow:
  pattern: critic_reviewer
  max_revisions: 3
  quality_threshold: 0.85

agents:
  # 生成者
  generator:
    id: content_creator
    role: generator
    specialization: content_writing
    config:
      max_concurrent_tasks: 2
      timeout_seconds: 300
    personality:
      personality: 85
      motivations: 90
      growth: 80
      emotions: 75
    capabilities:
      - creative_writing
      - research_integration
      - style_adaptation
      - seo_optimization

  # 批评者
  critic:
    id: content_critic
    role: critic
    specialization: content_critique
    config:
      max_concurrent_tasks: 2
    personality:
      personality: 70
      conflict: 90
      emotions: 65
    capabilities:
      - quality_assessment
      - bias_detection
      - clarity_analysis
      - engagement_evaluation

  # 验证者
  validator:
    id: content_validator
    role: validator
    validation_types:
      - grammar
      - factual_accuracy
      - brand_compliance
      - legal_compliance
    capabilities:
      - grammar_check
      - fact_verification
      - brand_guideline_check
      - legal_review

execution:
  generation:
    input:
      topic: "{{input.topic}}"
      target_audience: "{{input.target_audience}}"
      tone: "{{input.tone}}"
      length: "{{input.length}}"
      key_points: "{{input.key_points}}"
      seo_keywords: "{{input.seo_keywords}}"
      
    style_guide: "{{config.style_guide}}"
    brand_voice: "{{config.brand_voice}}"

  critique:
    criteria:
      - clarity
      - engagement
      - accuracy
      - originality
      - seo_effectiveness
      - brand_alignment
      
    feedback_format:
      - strength_points
      - improvement_areas
      - specific_suggestions
      - revision_priority

  validation:
    checks:
      - grammar_score >= 95
      - readability_score >= 60
      - seo_score >= 80
      - brand_compliance == 100
      - no_legal_flags

  revision:
    strategy: incremental
    preserve:
      - core_message
      - key_facts
      - approved_sections

output:
  formats:
    - markdown
    - html
    - plain_text
    
  metadata:
    - generation_timestamp
    - revision_count
    - quality_scores
    - critic_feedback_summary
    - validation_results

review:
  human_approval:
    required: "{{input.require_approval}}"
    reviewers: "{{input.approvers}}"
    
  auto_publish:
    enabled: "{{input.auto_publish}}"
    platforms: "{{input.publish_platforms}}"
    schedule: "{{input.publish_schedule}}"
```

## 6. 并行数据处理工作流

```yaml
# config/workflows/data_processing.yaml
name: data_processing
version: "2.0"
description: "大规模数据并行处理工作流 - 使用Parallel Processing模式"

workflow:
  pattern: parallel_processing
  min_success_rate: 0.9
  voting_mode: false

agents:
  # 数据分片处理器
  shards:
    - id: data_processor_1
      role: executor
      specialization: data_processing
      config:
        max_concurrent_tasks: 5
      capabilities:
        - etl_operations
        - data_transformation
        - quality_checks

    - id: data_processor_2
      role: executor
      specialization: data_processing
      config:
        max_concurrent_tasks: 5
      capabilities:
        - etl_operations
        - data_transformation
        - quality_checks

    - id: data_processor_3
      role: executor
      specialization: data_processing
      config:
        max_concurrent_tasks: 5
      capabilities:
        - etl_operations
        - data_transformation
        - quality_checks

    - id: data_processor_4
      role: executor
      specialization: data_processing
      config:
        max_concurrent_tasks: 5
      capabilities:
        - etl_operations
        - data_transformation
        - quality_checks

  # 结果聚合器
  aggregator:
    id: result_aggregator
    role: executor
    specialization: data_aggregation
    config:
      max_concurrent_tasks: 2
    capabilities:
      - result_consolidation
      - deduplication
      - final_validation

  # 验证者
  validator:
    id: data_validator
    role: validator
    validation_types:
      - data_integrity
      - schema_compliance
    capabilities:
      - row_count_verification
      - schema_validation
      - referential_integrity_check

execution:
  input:
    source: "{{input.data_source}}"
    format: "{{input.data_format}}"
    schema: "{{input.schema}}"
    
  sharding:
    strategy: hash_based
    shard_count: 4
    key: "{{input.shard_key}}"

  processing:
    steps:
      - extract
      - transform:
          operations: "{{input.transformations}}"
      - load:
          destination: "{{input.output_destination}}"
          
    quality_checks:
      - null_check
      - range_validation
      - format_validation
      - uniqueness_check

  aggregation:
    method: merge_sorted
    deduplication: true
    
    post_process:
      - global_sort
      - index_creation
      - statistics_calculation

  validation:
    final_checks:
      - row_count_match
      - schema_compliance
      - data_integrity

monitoring:
  metrics:
    - processing_rate
    - error_rate
    - queue_depth
    - memory_usage
    
  alerts:
    - condition: error_rate > 0.05
      severity: warning
      
    - condition: processing_rate < threshold
      severity: warning
      
    - condition: memory_usage > 0.9
      severity: critical

output:
  destination: "{{input.output_destination}}"
  format: "{{input.output_format}}"
  
  reports:
    - processing_summary
    - quality_report
    - performance_metrics
    - error_log
```

---

*配置文件版本: 1.0*  
*更新日期: 2026-02-27*  
*配套文档: AGENTS_v2.md, multi_agent_architecture.md, multi_agent_code_framework.md*
